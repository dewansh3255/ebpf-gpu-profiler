#!/usr/bin/env python3
"""
21_syscall_counter.py
Group 21 - GRS Project Part A

eBPF-based system call profiler. Hooks into the sys_enter and sys_exit
tracepoints to count syscalls by type and measure per-syscall latency.
Used to compare syscall overhead in containerized vs non-containerized
environments.

Usage:
    sudo python3 21_syscall_counter.py --duration 60 --output 21_syscall_results.csv
    sudo python3 21_syscall_counter.py --duration 60 --pid 12345

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


# eBPF program for syscall counting and latency measurement
BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>

// Per-syscall latency event
struct syscall_event_t {
    u32 pid;
    u32 tgid;
    u64 syscall_id;
    u64 latency_ns;
    char comm[16];
};

// Track entry timestamps: key = (pid, syscall_id)
struct entry_key_t {
    u32 pid;
    u64 syscall_id;
};

BPF_HASH(entry_ts, struct entry_key_t, u64);

// Per-syscall aggregate counters
struct syscall_stats_t {
    u64 count;
    u64 total_latency_ns;
    u64 min_latency_ns;
    u64 max_latency_ns;
};

BPF_HASH(syscall_stats, u64, struct syscall_stats_t);

// Perf output for per-event data (sampled)
BPF_PERF_OUTPUT(syscall_events);

// Total syscall counter
BPF_HASH(total_count, u32, u64);

// Optional PID filter (0 = no filter)
BPF_HASH(target_pid, u32, u32);

TRACEPOINT_PROBE(raw_syscalls, sys_enter)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    // Check PID filter
    u32 key = 0;
    u32 *tpid = target_pid.lookup(&key);
    if (tpid && *tpid != 0 && *tpid != pid)
        return 0;

    u64 ts = bpf_ktime_get_ns();
    u64 syscall_id = args->id;

    struct entry_key_t ekey = {};
    ekey.pid = pid;
    ekey.syscall_id = syscall_id;
    entry_ts.update(&ekey, &ts);

    return 0;
}

TRACEPOINT_PROBE(raw_syscalls, sys_exit)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    // Check PID filter
    u32 key = 0;
    u32 *tpid = target_pid.lookup(&key);
    if (tpid && *tpid != 0 && *tpid != pid)
        return 0;

    u64 ts = bpf_ktime_get_ns();
    u64 syscall_id = args->id;

    struct entry_key_t ekey = {};
    ekey.pid = pid;
    ekey.syscall_id = syscall_id;
    u64 *start = entry_ts.lookup(&ekey);
    if (!start)
        return 0;

    u64 latency = ts - *start;
    entry_ts.delete(&ekey);

    // Update aggregate stats
    struct syscall_stats_t *stats = syscall_stats.lookup(&syscall_id);
    if (stats) {
        stats->count += 1;
        stats->total_latency_ns += latency;
        if (latency < stats->min_latency_ns)
            stats->min_latency_ns = latency;
        if (latency > stats->max_latency_ns)
            stats->max_latency_ns = latency;
    } else {
        struct syscall_stats_t new_stats = {};
        new_stats.count = 1;
        new_stats.total_latency_ns = latency;
        new_stats.min_latency_ns = latency;
        new_stats.max_latency_ns = latency;
        syscall_stats.update(&syscall_id, &new_stats);
    }

    // Increment global counter
    u32 gkey = 0;
    u64 *cnt = total_count.lookup(&gkey);
    if (cnt) {
        *cnt += 1;
    } else {
        u64 one = 1;
        total_count.update(&gkey, &one);
    }

    return 0;
}
"""

# Common syscall number to name mapping (x86_64)
SYSCALL_NAMES = {
    0: "read", 1: "write", 2: "open", 3: "close", 4: "stat",
    5: "fstat", 6: "lstat", 7: "poll", 8: "lseek", 9: "mmap",
    10: "mprotect", 11: "munmap", 12: "brk", 13: "rt_sigaction",
    14: "rt_sigprocmask", 15: "rt_sigreturn", 16: "ioctl",
    17: "pread64", 18: "pwrite64", 19: "readv", 20: "writev",
    21: "access", 22: "pipe", 23: "select", 24: "sched_yield",
    25: "mremap", 28: "madvise", 32: "dup", 33: "dup2",
    35: "nanosleep", 39: "getpid", 41: "socket", 42: "connect",
    43: "accept", 44: "sendto", 45: "recvfrom", 46: "sendmsg",
    47: "recvmsg", 48: "shutdown", 49: "bind", 50: "listen",
    56: "clone", 57: "fork", 58: "vfork", 59: "execve",
    60: "exit", 61: "wait4", 62: "kill", 72: "fcntl",
    78: "getdents", 79: "getcwd", 80: "chdir", 82: "rename",
    83: "mkdir", 84: "rmdir", 87: "unlink", 89: "readlink",
    96: "gettimeofday", 102: "getuid", 110: "getppid",
    157: "prctl", 186: "gettid", 202: "futex", 217: "getdents64",
    228: "clock_gettime", 231: "exit_group", 232: "epoll_wait",
    233: "epoll_ctl", 257: "openat", 262: "newfstatat",
    272: "unshare", 281: "epoll_pwait", 288: "accept4",
    290: "eventfd2", 291: "epoll_create1", 292: "dup3",
    293: "pipe2", 302: "prlimit64", 318: "getrandom",
    332: "statx", 334: "rseq", 435: "clone3",
}


class SyscallStats:
    """Stores aggregate syscall statistics."""

    def __init__(self):
        self.count = 0
        self.total_latency = 0
        self.min_latency = float("inf")
        self.max_latency = 0


class SyscallCounter:
    """eBPF-based syscall counter and latency profiler."""

    def __init__(self, duration, output_file, target_pid=None):
        self.duration = duration
        self.output_file = output_file
        self.target_pid = target_pid or 0
        self.running = True

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.running = False

    def _get_syscall_name(self, nr):
        return SYSCALL_NAMES.get(nr, f"syscall_{nr}")

    def run(self):
        print(f"[Syscall Counter] Loading eBPF program...")
        b = BPF(text=BPF_PROGRAM)

        # Set PID filter if specified
        if self.target_pid > 0:
            key = ct.c_uint32(0)
            val = ct.c_uint32(self.target_pid)
            b["target_pid"][key] = val
            print(f"[Syscall Counter] Filtering to PID: {self.target_pid}")

        print(f"[Syscall Counter] Profiling for {self.duration} seconds...")
        start = time.time()

        while self.running and (time.time() - start) < self.duration:
            try:
                time.sleep(0.5)
            except KeyboardInterrupt:
                break

        elapsed = time.time() - start
        print(f"\n[Syscall Counter] Stopped after {elapsed:.1f} seconds.")

        # Collect results from eBPF maps
        results = self._collect_results(b)

        # Print summary
        self._print_summary(results, elapsed)

        # Save to CSV
        self._save_results(results, elapsed)

        return results

    def _collect_results(self, b):
        results = {}
        for syscall_id, stats in b["syscall_stats"].items():
            nr = syscall_id.value
            name = self._get_syscall_name(nr)
            results[name] = {
                "syscall_nr": nr,
                "name": name,
                "count": stats.count,
                "total_latency_ns": stats.total_latency_ns,
                "min_latency_ns": stats.min_latency_ns,
                "max_latency_ns": stats.max_latency_ns,
                "avg_latency_ns": (stats.total_latency_ns / stats.count
                                   if stats.count > 0 else 0),
            }
        return results

    def _print_summary(self, results, elapsed):
        print("\n" + "=" * 80)
        print("SYSCALL PROFILING SUMMARY")
        print("=" * 80)

        total_syscalls = sum(r["count"] for r in results.values())
        print(f"\nTotal syscalls: {total_syscalls}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Rate: {total_syscalls / elapsed:.0f} syscalls/sec")

        # Table header
        print(f"\n{'Syscall':<20} {'Count':>10} {'Rate/s':>10} "
              f"{'Avg(us)':>10} {'Min(us)':>10} {'Max(us)':>10}")
        print("-" * 80)

        sorted_results = sorted(results.values(),
                                key=lambda x: -x["count"])
        for r in sorted_results[:30]:
            print(f"{r['name']:<20} {r['count']:>10} "
                  f"{r['count'] / elapsed:>10.0f} "
                  f"{r['avg_latency_ns'] / 1000:>10.1f} "
                  f"{r['min_latency_ns'] / 1000:>10.1f} "
                  f"{r['max_latency_ns'] / 1000:>10.1f}")

    def _save_results(self, results, elapsed):
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "syscall_nr", "name", "count", "total_latency_ns",
                "avg_latency_ns", "min_latency_ns", "max_latency_ns"
            ])
            writer.writeheader()
            for r in sorted(results.values(), key=lambda x: -x["count"]):
                writer.writerow(r)

        print(f"\n[Syscall Counter] Results saved to: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="eBPF Syscall Counter and Latency Profiler - Group 21"
    )
    parser.add_argument("--duration", type=int, default=60,
                        help="Profiling duration in seconds (default: 60)")
    parser.add_argument("--output", type=str,
                        default="21_syscall_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--pid", type=int, default=None,
                        help="Filter by PID (optional)")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERROR: This script requires root privileges.")
        print("Run with: sudo python3 21_syscall_counter.py")
        sys.exit(1)

    counter = SyscallCounter(
        duration=args.duration,
        output_file=args.output,
        target_pid=args.pid,
    )
    counter.run()


if __name__ == "__main__":
    main()
