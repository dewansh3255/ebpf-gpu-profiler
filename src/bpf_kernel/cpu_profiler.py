#!/usr/bin/env python3
"""
21_cpu_profiler.py
Group 21 - GRS Project Part A

eBPF-based CPU profiler for measuring scheduling latency, context switch
frequency, and CPU utilization per process. Attaches to kernel scheduler
tracepoints to capture fine-grained CPU scheduling events.

Usage:
    sudo python3 21_cpu_profiler.py --duration 60 --output 21_cpu_results.csv

Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma
"""

import argparse
import csv
import ctypes as ct
import os
import signal
import sys
import time
from collections import defaultdict

try:
    from bcc import BPF
except ImportError:
    print("ERROR: BCC (BPF Compiler Collection) is not installed.")
    print("Install with: sudo apt-get install bpfcc-tools python3-bpfcc")
    sys.exit(1)


# eBPF program for CPU scheduling profiling
BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// Data structure for context switch events
struct ctx_switch_event_t {
    u32 prev_pid;
    u32 next_pid;
    u64 timestamp;
    u64 prev_runtime;       // how long prev process ran (ns)
    char prev_comm[16];
    char next_comm[16];
    u32 prev_state;
};

// Data structure for scheduling latency
struct sched_latency_t {
    u32 pid;
    u64 latency_ns;         // time spent waiting in run queue
    char comm[16];
};

// Hash map: pid -> timestamp when process became runnable
BPF_HASH(start_ts, u32, u64);

// Hash map: pid -> timestamp when process started running
BPF_HASH(run_start, u32, u64);

// Perf output buffers
BPF_PERF_OUTPUT(ctx_switch_events);
BPF_PERF_OUTPUT(sched_latency_events);

// Counters
BPF_HASH(ctx_switch_count, u32, u64);    // per-CPU context switch count
BPF_HASH(total_runtime, u32, u64);       // per-PID total runtime (ns)

// Tracepoint: sched_wakeup - process becomes runnable
TRACEPOINT_PROBE(sched, sched_wakeup)
{
    u32 pid = args->pid;
    u64 ts = bpf_ktime_get_ns();
    start_ts.update(&pid, &ts);
    return 0;
}

// Tracepoint: sched_wakeup_new - new process becomes runnable
TRACEPOINT_PROBE(sched, sched_wakeup_new)
{
    u32 pid = args->pid;
    u64 ts = bpf_ktime_get_ns();
    start_ts.update(&pid, &ts);
    return 0;
}

// Tracepoint: sched_switch - context switch occurs
TRACEPOINT_PROBE(sched, sched_switch)
{
    u64 ts = bpf_ktime_get_ns();
    u32 prev_pid = args->prev_pid;
    u32 next_pid = args->next_pid;

    // --- Handle the outgoing process (prev) ---
    // Calculate how long it ran
    u64 *rs = run_start.lookup(&prev_pid);
    u64 runtime = 0;
    if (rs) {
        runtime = ts - *rs;
        // Accumulate total runtime
        u64 *existing = total_runtime.lookup(&prev_pid);
        if (existing) {
            *existing += runtime;
        } else {
            total_runtime.update(&prev_pid, &runtime);
        }
        run_start.delete(&prev_pid);
    }

    // --- Handle the incoming process (next) ---
    // Record when it starts running
    run_start.update(&next_pid, &ts);

    // Calculate scheduling latency (time in run queue)
    u64 *wakeup_ts = start_ts.lookup(&next_pid);
    if (wakeup_ts) {
        u64 latency = ts - *wakeup_ts;
        struct sched_latency_t lat_event = {};
        lat_event.pid = next_pid;
        lat_event.latency_ns = latency;
        bpf_get_current_comm(&lat_event.comm, sizeof(lat_event.comm));
        sched_latency_events.perf_submit(args, &lat_event, sizeof(lat_event));
        start_ts.delete(&next_pid);
    }

    // --- Emit context switch event ---
    struct ctx_switch_event_t event = {};
    event.prev_pid = prev_pid;
    event.next_pid = next_pid;
    event.timestamp = ts;
    event.prev_runtime = runtime;
    event.prev_state = args->prev_state;
    bpf_probe_read_kernel_str(&event.prev_comm, sizeof(event.prev_comm),
                              args->prev_comm);
    bpf_probe_read_kernel_str(&event.next_comm, sizeof(event.next_comm),
                              args->next_comm);
    ctx_switch_events.perf_submit(args, &event, sizeof(event));

    // Increment per-CPU context switch counter
    u32 cpu = bpf_get_smp_processor_id();
    u64 *count = ctx_switch_count.lookup(&cpu);
    if (count) {
        *count += 1;
    } else {
        u64 one = 1;
        ctx_switch_count.update(&cpu, &one);
    }

    return 0;
}
"""


class CtxSwitchEvent(ct.Structure):
    """Mirrors struct ctx_switch_event_t from eBPF."""
    _fields_ = [
        ("prev_pid", ct.c_uint32),
        ("next_pid", ct.c_uint32),
        ("timestamp", ct.c_uint64),
        ("prev_runtime", ct.c_uint64),
        ("prev_comm", ct.c_char * 16),
        ("next_comm", ct.c_char * 16),
        ("prev_state", ct.c_uint32),
    ]


class SchedLatencyEvent(ct.Structure):
    """Mirrors struct sched_latency_t from eBPF."""
    _fields_ = [
        ("pid", ct.c_uint32),
        ("latency_ns", ct.c_uint64),
        ("comm", ct.c_char * 16),
    ]


class CPUProfiler:
    """eBPF-based CPU scheduling profiler."""

    def __init__(self, duration, output_file, target_comm=None):
        self.duration = duration
        self.output_file = output_file
        self.target_comm = target_comm
        self.running = True
        self.ctx_switches = []
        self.sched_latencies = []
        self.start_time = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.running = False

    def _ctx_switch_callback(self, cpu, data, size):
        event = ct.cast(data, ct.POINTER(CtxSwitchEvent)).contents
        self.ctx_switches.append({
            "timestamp_ns": event.timestamp,
            "prev_pid": event.prev_pid,
            "next_pid": event.next_pid,
            "prev_comm": event.prev_comm.decode("utf-8", errors="replace"),
            "next_comm": event.next_comm.decode("utf-8", errors="replace"),
            "prev_runtime_ns": event.prev_runtime,
            "prev_state": event.prev_state,
        })

    def _latency_callback(self, cpu, data, size):
        event = ct.cast(data, ct.POINTER(SchedLatencyEvent)).contents
        self.sched_latencies.append({
            "timestamp_ns": time.time_ns(),
            "pid": event.pid,
            "comm": event.comm.decode("utf-8", errors="replace"),
            "latency_ns": event.latency_ns,
        })

    def run(self):
        print(f"[CPU Profiler] Loading eBPF program...")
        b = BPF(text=BPF_PROGRAM)

        b["ctx_switch_events"].open_perf_buffer(self._ctx_switch_callback,
                                                 page_cnt=64)
        b["sched_latency_events"].open_perf_buffer(self._latency_callback,
                                                    page_cnt=64)

        print(f"[CPU Profiler] Profiling for {self.duration} seconds...")
        print(f"[CPU Profiler] Press Ctrl+C to stop early.")
        self.start_time = time.time()

        while self.running and (time.time() - self.start_time) < self.duration:
            try:
                b.perf_buffer_poll(timeout=100)
            except KeyboardInterrupt:
                break

        elapsed = time.time() - self.start_time
        print(f"\n[CPU Profiler] Stopped after {elapsed:.1f} seconds.")

        # Print summary
        self._print_summary(b)

        # Save results
        self._save_results()

        return self.ctx_switches, self.sched_latencies

    def _print_summary(self, b):
        print("\n" + "=" * 60)
        print("CPU PROFILING SUMMARY")
        print("=" * 60)

        # Context switch counts per CPU
        print("\nContext Switches per CPU:")
        ctx_counts = b["ctx_switch_count"]
        total = 0
        for cpu_id, count in sorted(ctx_counts.items(),
                                     key=lambda x: x[0].value):
            print(f"  CPU {cpu_id.value}: {count.value} switches")
            total += count.value
        print(f"  Total: {total} switches")
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            print(f"  Rate: {total / elapsed:.0f} switches/sec")

        # Scheduling latency statistics
        if self.sched_latencies:
            latencies = [e["latency_ns"] for e in self.sched_latencies]
            latencies.sort()
            n = len(latencies)
            print(f"\nScheduling Latency (run-queue wait time):")
            print(f"  Samples: {n}")
            print(f"  Min:     {latencies[0] / 1000:.1f} us")
            print(f"  Median:  {latencies[n // 2] / 1000:.1f} us")
            print(f"  P95:     {latencies[int(n * 0.95)] / 1000:.1f} us")
            print(f"  P99:     {latencies[int(n * 0.99)] / 1000:.1f} us")
            print(f"  Max:     {latencies[-1] / 1000:.1f} us")
            print(f"  Mean:    {sum(latencies) / n / 1000:.1f} us")

        # Top processes by context switches
        proc_switches = defaultdict(int)
        for e in self.ctx_switches:
            proc_switches[e["next_comm"]] += 1
        top_procs = sorted(proc_switches.items(), key=lambda x: -x[1])[:10]
        print(f"\nTop 10 Processes by Context Switches:")
        for comm, count in top_procs:
            print(f"  {comm:20s} {count:>8d}")

    def _save_results(self):
        # Save context switch data
        cs_file = self.output_file.replace(".csv", "_ctx_switches.csv")
        with open(cs_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp_ns", "prev_pid", "next_pid",
                "prev_comm", "next_comm", "prev_runtime_ns", "prev_state"
            ])
            writer.writeheader()
            writer.writerows(self.ctx_switches)
        print(f"\n[CPU Profiler] Context switches saved to: {cs_file}")
        print(f"  ({len(self.ctx_switches)} records)")

        # Save scheduling latency data
        lat_file = self.output_file.replace(".csv", "_sched_latency.csv")
        with open(lat_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp_ns", "pid", "comm", "latency_ns"
            ])
            writer.writeheader()
            writer.writerows(self.sched_latencies)
        print(f"[CPU Profiler] Scheduling latencies saved to: {lat_file}")
        print(f"  ({len(self.sched_latencies)} records)")


def main():
    parser = argparse.ArgumentParser(
        description="eBPF CPU Scheduler Profiler - Group 21"
    )
    parser.add_argument("--duration", type=int, default=60,
                        help="Profiling duration in seconds (default: 60)")
    parser.add_argument("--output", type=str, default="21_cpu_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--filter-comm", type=str, default=None,
                        help="Filter by process command name (optional)")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERROR: This script requires root privileges.")
        print("Run with: sudo python3 21_cpu_profiler.py")
        sys.exit(1)

    profiler = CPUProfiler(
        duration=args.duration,
        output_file=args.output,
        target_comm=args.filter_comm,
    )
    profiler.run()


if __name__ == "__main__":
    main()
