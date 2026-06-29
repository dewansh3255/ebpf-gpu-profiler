#!/usr/bin/env python3
"""
llc_profiler.py
Last-Level-Cache (LLC) miss profiler using the CPU Performance Monitoring Unit
(PMU) via eBPF perf-event hooks.

Why: containerisation adds namespace/cgroup machinery and extra context
switching. Every context switch and cross-namespace transition can evict a
process's hot working set from the shared L3/LLC, forcing slow DRAM refills.
Comparing LLC miss behaviour of the *same* ML workload under bare-metal vs
Docker exposes this hidden "cache pollution" tax that wall-clock timing alone
hides.

How: we open two hardware PMU counters through `perf_event_open` (driven by
BCC's attach_perf_event) sampling on overflow:
  - LLC misses      (PERF_COUNT_HW_CACHE_LL, OP_READ, RESULT_MISS)
  - LLC references  (PERF_COUNT_HW_CACHE_LL, OP_READ, RESULT_ACCESS)
Each time a counter overflows by `sample_period` events, the kernel fires our
eBPF program on the CPU that overflowed. The program attributes that overflow
to the currently-running pid/comm and its cgroup id, accumulating per-process
sample counts. Estimated absolute misses = samples * sample_period.

Tracking the cgroup id lets us distinguish container processes (non-root
cgroup) from host processes.

Requires hardware PMU access (bare metal or a VM/cloud instance with vPMU
exposed). On platforms without LLC PMU support the counter open will fail and
we report that clearly rather than producing bogus data.

Usage:
    sudo env PYTHONPATH=/usr/lib/python3/dist-packages python3 \
        llc_profiler.py --duration 120 --sample-period 10000 \
        --output results/native/llc.csv
"""

import argparse
import csv
import os
import signal
import sys
import time

try:
    from bcc import BPF, PerfType
except ImportError:
    print("ERROR: BCC is not installed. Install: sudo apt-get install python3-bpfcc")
    sys.exit(1)

# PerfHWCacheConfig is not exported by all BCC versions; fall back to the
# raw perf_event_open cache-config constants if it is missing.
try:
    from bcc import PerfHWCacheConfig
    _LL = PerfHWCacheConfig.LL
    _OP_READ = PerfHWCacheConfig.OP_READ
    _RESULT_ACCESS = PerfHWCacheConfig.RESULT_ACCESS
    _RESULT_MISS = PerfHWCacheConfig.RESULT_MISS
except ImportError:
    _LL = 2            # PERF_COUNT_HW_CACHE_LL
    _OP_READ = 0       # PERF_COUNT_HW_CACHE_OP_READ
    _RESULT_ACCESS = 0 # PERF_COUNT_HW_CACHE_RESULT_ACCESS
    _RESULT_MISS = 1   # PERF_COUNT_HW_CACHE_RESULT_MISS


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <uapi/linux/bpf_perf_event.h>

struct key_t {
    u32 pid;
    char comm[16];
};

// per-process accumulated overflow samples
BPF_HASH(llc_miss_samples, struct key_t, u64);
BPF_HASH(llc_ref_samples, struct key_t, u64);
// per-process cgroup id (container detection)
BPF_HASH(cgroup_of, u32, u64);

static inline void fill_key(struct key_t *k) {
    k->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&k->comm, sizeof(k->comm));
}

int on_llc_miss(struct bpf_perf_event_data *ctx) {
    struct key_t k = {};
    fill_key(&k);
    llc_miss_samples.increment(k);
    u64 cg = bpf_get_current_cgroup_id();
    cgroup_of.update(&k.pid, &cg);
    return 0;
}

int on_llc_ref(struct bpf_perf_event_data *ctx) {
    struct key_t k = {};
    fill_key(&k);
    llc_ref_samples.increment(k);
    return 0;
}
"""


class LLCProfiler:
    def __init__(self, duration, sample_period, output_file):
        self.duration = duration
        self.sample_period = sample_period
        self.output_file = output_file
        self.running = True
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, *_):
        self.running = False

    def _llc_config(self, result):
        # encode PERF_TYPE_HW_CACHE config: cache | (op << 8) | (result << 16)
        return (_LL | (_OP_READ << 8) | (result << 16))

    def run(self):
        if os.geteuid() != 0:
            print("ERROR: PMU perf events require root. Re-run with sudo.")
            sys.exit(1)

        print("[LLC Profiler] Compiling eBPF program...")
        b = BPF(text=BPF_PROGRAM)

        miss_cfg = self._llc_config(_RESULT_MISS)
        ref_cfg = self._llc_config(_RESULT_ACCESS)

        try:
            b.attach_perf_event(
                ev_type=PerfType.HW_CACHE, ev_config=miss_cfg,
                fn_name="on_llc_miss", sample_period=self.sample_period)
            b.attach_perf_event(
                ev_type=PerfType.HW_CACHE, ev_config=ref_cfg,
                fn_name="on_llc_ref", sample_period=self.sample_period)
        except Exception as e:
            print("[LLC Profiler] FATAL: could not open LLC PMU counters.")
            print(f"  Reason: {e}")
            print("  This host likely lacks LLC PMU support (e.g. a VM without "
                  "vPMU). Run on bare metal or enable the virtual PMU.")
            sys.exit(2)

        print(f"[LLC Profiler] Sampling LLC misses/refs every "
              f"{self.sample_period} events for {self.duration}s. Ctrl+C to stop.")

        start = time.time()
        while self.running and (time.time() - start) < self.duration:
            try:
                time.sleep(0.5)
            except KeyboardInterrupt:
                break
        elapsed = time.time() - start

        self._collect_and_save(b, elapsed)

    def _collect_and_save(self, b, elapsed):
        miss = b["llc_miss_samples"]
        ref = b["llc_ref_samples"]
        cg = b["cgroup_of"]

        # aggregate by (pid, comm)
        rows = {}
        for k, v in miss.items():
            comm = k.comm.decode("utf-8", errors="replace")
            rows[(k.pid, comm)] = {
                "pid": k.pid, "comm": comm,
                "llc_misses": v.value * self.sample_period,
                "llc_refs": 0,
            }
        for k, v in ref.items():
            comm = k.comm.decode("utf-8", errors="replace")
            key = (k.pid, comm)
            row = rows.setdefault(key, {
                "pid": k.pid, "comm": comm, "llc_misses": 0, "llc_refs": 0})
            row["llc_refs"] = v.value * self.sample_period

        cg_map = {k.value: v.value for k, v in cg.items()}

        out = []
        for (pid, comm), row in rows.items():
            refs = row["llc_refs"]
            misses = row["llc_misses"]
            miss_rate = (100.0 * misses / refs) if refs > 0 else 0.0
            cgid = cg_map.get(pid, 0)
            out.append({
                "pid": pid,
                "comm": comm,
                "cgroup_id": cgid,
                "containerized": int(cgid not in (0, 1)),
                "llc_misses_est": misses,
                "llc_refs_est": refs,
                "llc_miss_rate_pct": round(miss_rate, 2),
                "misses_per_sec": round(misses / elapsed, 0) if elapsed > 0 else 0,
            })
        out.sort(key=lambda r: -r["llc_misses_est"])

        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)),
                    exist_ok=True)
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pid", "comm", "cgroup_id", "containerized",
                "llc_misses_est", "llc_refs_est", "llc_miss_rate_pct",
                "misses_per_sec",
            ])
            writer.writeheader()
            writer.writerows(out)
        print(f"\n[LLC Profiler] Saved -> {self.output_file}")

        # summary
        total_miss = sum(r["llc_misses_est"] for r in out)
        total_ref = sum(r["llc_refs_est"] for r in out)
        overall_rate = (100.0 * total_miss / total_ref) if total_ref else 0
        print("\n" + "=" * 72)
        print("LLC SUMMARY (estimated; sample_period = "
              f"{self.sample_period})")
        print("=" * 72)
        print(f"Total LLC misses ~{total_miss:,}  refs ~{total_ref:,}  "
              f"overall miss rate {overall_rate:.2f}%")
        print(f"\n{'comm':<18}{'pid':>8}{'misses':>16}{'miss%':>9}{'ctr':>5}")
        for r in out[:15]:
            print(f"{r['comm']:<18}{r['pid']:>8}{r['llc_misses_est']:>16,}"
                  f"{r['llc_miss_rate_pct']:>9.2f}{r['containerized']:>5}")


def main():
    parser = argparse.ArgumentParser(description="eBPF PMU LLC miss profiler")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--sample-period", type=int, default=10000,
                        help="PMU overflow period in events (default: 10000)")
    parser.add_argument("--output", type=str, default="results/llc.csv")
    args = parser.parse_args()

    LLCProfiler(args.duration, args.sample_period, args.output).run()


if __name__ == "__main__":
    main()
