#!/usr/bin/env python3
"""
cuda_uprobe_monitor.py
eBPF uprobe tracer for the NVIDIA CUDA user-space driver (libcuda.so).

This is the "true execution probe". It separates two fundamentally different
costs that a naive launch-only trace conflates:

  1. Driver-queue / launch overhead (CPU-side):
     `cuLaunchKernel` only *enqueues* work onto a CUDA stream and returns almost
     immediately. The entry->return duration of that call is pure CPU-side
     driver bookkeeping -- it says nothing about how long the GPU actually ran.
     Logged as event_type = LAUNCH_QUEUE.

  2. Hardware execution + sync latency:
     `cuStreamSynchronize` (and `cuCtxSynchronize`) blocks the calling thread
     until all previously queued work on the stream has finished on the device.
     Its entry->return duration is the wall-clock time the CPU spends waiting on
     the GPU, i.e. real hardware execution + queue drain + sync overhead.
     Logged as event_type = HW_EXEC_SYNC.

By attaching uprobes/uretprobes to BOTH symbols we can attribute time correctly:
container vs bare-metal differences in LAUNCH_QUEUE point to CPU/driver/syscall
overhead, while differences in HW_EXEC_SYNC point to the device side.

We also keep the host<->device memory transfer probes (MEM_TRANSFER) from the
original implementation.

Every row carries the event-completion timestamp on CLOCK_MONOTONIC
(bpf_ktime_get_ns), the call duration, and pid/tid/comm so the analysis layer
can reconstruct per-thread launch/sync timelines and compute GPU idle gaps.

Usage:
    sudo env PYTHONPATH=/usr/lib/python3/dist-packages python3 \
        cuda_uprobe_monitor.py --duration 120 --output results/native/cuda_trace.csv
"""

import argparse
import csv
import ctypes as ct
import os
import signal
import sys
import time

try:
    from bcc import BPF
except ImportError:
    print("ERROR: BCC is not installed. Install: sudo apt-get install python3-bpfcc")
    sys.exit(1)


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>

#define EVT_LAUNCH_QUEUE  0
#define EVT_HW_EXEC_SYNC  1
#define EVT_MEM_TRANSFER  2

struct gpu_event_t {
    u64 ts_ns;          // event completion (uretprobe) timestamp, CLOCK_MONOTONIC
    u64 duration_ns;    // entry->return call duration
    u64 gap_ns;         // for SYNC: time since last kernel launch return on this tid
    u32 pid;
    u32 tid;
    u8  event_type;
    char comm[16];
};
BPF_PERF_OUTPUT(gpu_events);

// per-tid entry timestamps
BPF_HASH(launch_start, u32, u64);
BPF_HASH(sync_start, u32, u64);
BPF_HASH(mem_start, u32, u64);
// per-tid timestamp of the most recent cuLaunchKernel return (for sync gap)
BPF_HASH(last_launch_ret, u32, u64);

// ---- cuLaunchKernel: CPU-side enqueue overhead ----
int probe_launch_entry(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    launch_start.update(&tid, &ts);
    return 0;
}
int probe_launch_return(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 *tsp = launch_start.lookup(&tid);
    if (!tsp) return 0;
    u64 now = bpf_ktime_get_ns();

    struct gpu_event_t e = {};
    e.ts_ns = now;
    e.duration_ns = now - *tsp;
    e.gap_ns = 0;
    e.pid = bpf_get_current_pid_tgid() >> 32;
    e.tid = tid;
    e.event_type = EVT_LAUNCH_QUEUE;
    bpf_get_current_comm(&e.comm, sizeof(e.comm));
    gpu_events.perf_submit(ctx, &e, sizeof(e));

    launch_start.delete(&tid);
    last_launch_ret.update(&tid, &now);
    return 0;
}

// ---- cuStreamSynchronize / cuCtxSynchronize: HW execution + sync latency ----
int probe_sync_entry(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    sync_start.update(&tid, &ts);
    return 0;
}
int probe_sync_return(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 *tsp = sync_start.lookup(&tid);
    if (!tsp) return 0;
    u64 now = bpf_ktime_get_ns();

    struct gpu_event_t e = {};
    e.ts_ns = now;
    e.duration_ns = now - *tsp;       // CPU blocked == HW exec + sync
    u64 *llr = last_launch_ret.lookup(&tid);
    e.gap_ns = llr ? (now - *llr) : 0;
    e.pid = bpf_get_current_pid_tgid() >> 32;
    e.tid = tid;
    e.event_type = EVT_HW_EXEC_SYNC;
    bpf_get_current_comm(&e.comm, sizeof(e.comm));
    gpu_events.perf_submit(ctx, &e, sizeof(e));

    sync_start.delete(&tid);
    return 0;
}

// ---- cuMemcpy* : PCIe host<->device transfers ----
int probe_mem_entry(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    mem_start.update(&tid, &ts);
    return 0;
}
int probe_mem_return(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 *tsp = mem_start.lookup(&tid);
    if (!tsp) return 0;
    u64 now = bpf_ktime_get_ns();

    struct gpu_event_t e = {};
    e.ts_ns = now;
    e.duration_ns = now - *tsp;
    e.gap_ns = 0;
    e.pid = bpf_get_current_pid_tgid() >> 32;
    e.tid = tid;
    e.event_type = EVT_MEM_TRANSFER;
    bpf_get_current_comm(&e.comm, sizeof(e.comm));
    gpu_events.perf_submit(ctx, &e, sizeof(e));

    mem_start.delete(&tid);
    return 0;
}
"""

EVENT_TYPES = {0: "LAUNCH_QUEUE", 1: "HW_EXEC_SYNC", 2: "MEM_TRANSFER"}


class GpuEvent(ct.Structure):
    _fields_ = [
        ("ts_ns", ct.c_uint64),
        ("duration_ns", ct.c_uint64),
        ("gap_ns", ct.c_uint64),
        ("pid", ct.c_uint32),
        ("tid", ct.c_uint32),
        ("event_type", ct.c_uint8),
        ("comm", ct.c_char * 16),
    ]


# Symbols to hook. Launch + sync are the key additions; memcpy variants vary by
# driver version so we attach whichever are present.
LAUNCH_SYMS = ["cuLaunchKernel", "cuLaunchKernelEx"]
SYNC_SYMS = ["cuStreamSynchronize", "cuCtxSynchronize"]
MEM_SYMS = [
    "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2", "cuMemcpyAsync",
    "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2",
]


class CudaUprobeMonitor:
    def __init__(self, duration, output_file, libcuda="cuda", max_events=1000000):
        self.duration = duration
        self.output_file = output_file
        self.libcuda = libcuda
        # Kernel launches fire thousands/sec during training; cap stored rows so
        # a long capture can't exhaust RAM. Summary percentiles below use only
        # the retained sample.
        self.max_events = max_events
        self.dropped = 0
        self.events = []
        self.running = True
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, *_):
        self.running = False

    def _attach(self, b, syms, entry_fn, ret_fn, label):
        hooked = []
        for sym in syms:
            try:
                b.attach_uprobe(name=self.libcuda, sym=sym, fn_name=entry_fn)
                b.attach_uretprobe(name=self.libcuda, sym=sym, fn_name=ret_fn)
                hooked.append(sym)
            except Exception:
                pass
        if hooked:
            print(f"[CUDA Monitor] {label}: attached {', '.join(hooked)}")
        else:
            print(f"[CUDA Monitor] WARNING: no {label} symbols found in libcuda")
        return hooked

    def _callback(self, cpu, data, size):
        if len(self.events) >= self.max_events:
            self.dropped += 1
            return
        e = ct.cast(data, ct.POINTER(GpuEvent)).contents
        self.events.append({
            "ts_mono_ns": e.ts_ns,
            "event_type": EVENT_TYPES.get(e.event_type, "unknown"),
            "duration_ns": e.duration_ns,
            "exec_gap_ns": e.gap_ns,
            "pid": e.pid,
            "tid": e.tid,
            "comm": e.comm.decode("utf-8", errors="replace"),
        })

    def run(self):
        if os.geteuid() != 0:
            print("ERROR: uprobes on libcuda require root. Re-run with sudo.")
            sys.exit(1)

        print("[CUDA Monitor] Compiling eBPF program...")
        b = BPF(text=BPF_PROGRAM)

        launch = self._attach(b, LAUNCH_SYMS, "probe_launch_entry",
                              "probe_launch_return", "LAUNCH_QUEUE")
        sync = self._attach(b, SYNC_SYMS, "probe_sync_entry",
                           "probe_sync_return", "HW_EXEC_SYNC")
        self._attach(b, MEM_SYMS, "probe_mem_entry",
                    "probe_mem_return", "MEM_TRANSFER")

        if not launch and not sync:
            print("[CUDA Monitor] FATAL: could not attach to libcuda.so. "
                  "Is the NVIDIA driver loaded?")
            sys.exit(1)

        b["gpu_events"].open_perf_buffer(self._callback, page_cnt=128)
        print(f"[CUDA Monitor] Tracing for {self.duration}s. Ctrl+C to stop.")

        start = time.time()
        while self.running and (time.time() - start) < self.duration:
            try:
                b.perf_buffer_poll(timeout=200)
            except KeyboardInterrupt:
                break

        print(f"\n[CUDA Monitor] Captured {len(self.events)} events.")
        self._save()
        self._print_summary()

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)),
                    exist_ok=True)
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ts_mono_ns", "event_type", "duration_ns", "exec_gap_ns",
                "pid", "tid", "comm",
            ])
            writer.writeheader()
            writer.writerows(self.events)
        print(f"[CUDA Monitor] Saved -> {self.output_file}")
        if self.dropped:
            print(f"[CUDA Monitor] NOTE: capped at {self.max_events} events; "
                  f"dropped {self.dropped} (summary uses the retained sample).")

    def _print_summary(self):
        if not self.events:
            return
        agg = {}
        for e in self.events:
            d = agg.setdefault(e["event_type"], [])
            d.append(e["duration_ns"])
        print("\n" + "=" * 60)
        print("CUDA DRIVER-API SUMMARY (us)")
        print("=" * 60)
        print(f"{'event':<16}{'count':>8}{'mean':>12}{'p50':>12}{'p99':>12}")
        for etype, durs in sorted(agg.items()):
            durs.sort()
            n = len(durs)
            mean = sum(durs) / n / 1000
            p50 = durs[n // 2] / 1000
            p99 = durs[min(n - 1, int(n * 0.99))] / 1000
            print(f"{etype:<16}{n:>8}{mean:>12.2f}{p50:>12.2f}{p99:>12.2f}")
        print("\nNote: HW_EXEC_SYNC = GPU execution + sync (CPU blocked); "
              "LAUNCH_QUEUE = CPU-side driver enqueue overhead.")


def main():
    parser = argparse.ArgumentParser(
        description="CUDA driver-API uprobe monitor (launch queue vs HW sync)")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--output", type=str, default="results/cuda_trace.csv")
    parser.add_argument("--libcuda", type=str, default="cuda",
                        help="library name/path for libcuda (default: 'cuda')")
    parser.add_argument("--max-events", type=int, default=1000000,
                        help="Cap on stored events to bound memory (default: 1M)")
    args = parser.parse_args()

    CudaUprobeMonitor(args.duration, args.output, args.libcuda,
                      max_events=args.max_events).run()


if __name__ == "__main__":
    main()
