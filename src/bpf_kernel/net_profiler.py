#!/usr/bin/env python3
"""
21_net_profiler.py
Group 21 - GRS Project Part A

eBPF-based network stack profiler. Traces TCP send/receive operations
and network device transmissions to measure network processing time.
Critical for comparing containerized (veth + bridge) vs native networking
overhead in multi-GPU ML workloads using NCCL.

Usage:
    sudo python3 21_net_profiler.py --duration 60 --output 21_net_results.csv

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


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>



// Event structure for network operations
struct net_event_t {
    u32 pid;
    u64 timestamp;
    u64 latency_ns;
    u32 bytes;
    u8  event_type;     // 0 = tcp_send, 1 = tcp_recv, 2 = net_xmit
    char comm[16];
};

// Track function entry timestamps
BPF_HASH(send_start, u32, u64);
BPF_HASH(recv_start, u32, u64);

// Aggregate stats
struct net_stats_t {
    u64 count;
    u64 total_bytes;
    u64 total_latency_ns;
    u64 min_latency_ns;
    u64 max_latency_ns;
};

BPF_HASH(tcp_send_stats, u32, struct net_stats_t);   // key = 0
BPF_HASH(tcp_recv_stats, u32, struct net_stats_t);   // key = 0
BPF_HASH(net_xmit_count, u32, u64);

BPF_PERF_OUTPUT(net_events);

// --- TCP sendmsg ---
int trace_tcp_sendmsg(struct pt_regs *ctx, void *sk,
                       void *msg, size_t size)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 ts = bpf_ktime_get_ns();
    send_start.update(&pid, &ts);
    return 0;
}

int trace_tcp_sendmsg_ret(struct pt_regs *ctx)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 ts = bpf_ktime_get_ns();
    u64 *start = send_start.lookup(&pid);
    if (!start)
        return 0;

    u64 latency = ts - *start;
    send_start.delete(&pid);

    int ret = PT_REGS_RC(ctx);
    u32 bytes = ret > 0 ? (u32)ret : 0;

    // Update aggregate stats
    u32 key = 0;
    struct net_stats_t *stats = tcp_send_stats.lookup(&key);
    if (stats) {
        stats->count += 1;
        stats->total_bytes += bytes;
        stats->total_latency_ns += latency;
        if (latency < stats->min_latency_ns)
            stats->min_latency_ns = latency;
        if (latency > stats->max_latency_ns)
            stats->max_latency_ns = latency;
    } else {
        struct net_stats_t new_stats = {};
        new_stats.count = 1;
        new_stats.total_bytes = bytes;
        new_stats.total_latency_ns = latency;
        new_stats.min_latency_ns = latency;
        new_stats.max_latency_ns = latency;
        tcp_send_stats.update(&key, &new_stats);
    }

    // Emit event
    struct net_event_t event = {};
    event.pid = pid;
    event.timestamp = ts;
    event.latency_ns = latency;
    event.bytes = bytes;
    event.event_type = 0;  // tcp_send
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    net_events.perf_submit(ctx, &event, sizeof(event));

    return 0;
}

// --- TCP recvmsg ---
int trace_tcp_recvmsg(struct pt_regs *ctx)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 ts = bpf_ktime_get_ns();
    recv_start.update(&pid, &ts);
    return 0;
}

int trace_tcp_recvmsg_ret(struct pt_regs *ctx)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 ts = bpf_ktime_get_ns();
    u64 *start = recv_start.lookup(&pid);
    if (!start)
        return 0;

    u64 latency = ts - *start;
    recv_start.delete(&pid);

    int ret = PT_REGS_RC(ctx);
    u32 bytes = ret > 0 ? (u32)ret : 0;

    // Update aggregate stats
    u32 key = 0;
    struct net_stats_t *stats = tcp_recv_stats.lookup(&key);
    if (stats) {
        stats->count += 1;
        stats->total_bytes += bytes;
        stats->total_latency_ns += latency;
        if (latency < stats->min_latency_ns)
            stats->min_latency_ns = latency;
        if (latency > stats->max_latency_ns)
            stats->max_latency_ns = latency;
    } else {
        struct net_stats_t new_stats = {};
        new_stats.count = 1;
        new_stats.total_bytes = bytes;
        new_stats.total_latency_ns = latency;
        new_stats.min_latency_ns = latency;
        new_stats.max_latency_ns = latency;
        tcp_recv_stats.update(&key, &new_stats);
    }

    // Emit event
    struct net_event_t event = {};
    event.pid = pid;
    event.timestamp = ts;
    event.latency_ns = latency;
    event.bytes = bytes;
    event.event_type = 1;  // tcp_recv
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    net_events.perf_submit(ctx, &event, sizeof(event));

    return 0;
}

// --- Network device transmit ---
TRACEPOINT_PROBE(net, net_dev_xmit)
{
    u32 key = 0;
    u64 *cnt = net_xmit_count.lookup(&key);
    if (cnt) {
        *cnt += 1;
    } else {
        u64 one = 1;
        net_xmit_count.update(&key, &one);
    }
    return 0;
}
"""


class NetEvent(ct.Structure):
    _fields_ = [
        ("pid", ct.c_uint32),
        ("timestamp", ct.c_uint64),
        ("latency_ns", ct.c_uint64),
        ("bytes", ct.c_uint32),
        ("event_type", ct.c_uint8),
        ("comm", ct.c_char * 16),
    ]


EVENT_TYPES = {0: "tcp_send", 1: "tcp_recv", 2: "net_xmit"}


class NetworkProfiler:
    """eBPF network stack profiler."""

    def __init__(self, duration, output_file):
        self.duration = duration
        self.output_file = output_file
        self.running = True
        self.events = []

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.running = False

    def _event_callback(self, cpu, data, size):
        event = ct.cast(data, ct.POINTER(NetEvent)).contents
        self.events.append({
            "timestamp_ns": event.timestamp,
            "pid": event.pid,
            "comm": event.comm.decode("utf-8", errors="replace"),
            "event_type": EVENT_TYPES.get(event.event_type, "unknown"),
            "latency_ns": event.latency_ns,
            "bytes": event.bytes,
        })

    def run(self):
        print(f"[Network Profiler] Loading eBPF program...")
        b = BPF(text=BPF_PROGRAM)

        # Attach kprobes to TCP functions
        b.attach_kprobe(event="tcp_sendmsg", fn_name="trace_tcp_sendmsg")
        b.attach_kretprobe(event="tcp_sendmsg",
                           fn_name="trace_tcp_sendmsg_ret")
        b.attach_kprobe(event="tcp_recvmsg", fn_name="trace_tcp_recvmsg")
        b.attach_kretprobe(event="tcp_recvmsg",
                           fn_name="trace_tcp_recvmsg_ret")

        b["net_events"].open_perf_buffer(self._event_callback, page_cnt=64)

        print(f"[Network Profiler] Profiling for {self.duration} seconds...")
        start = time.time()

        while self.running and (time.time() - start) < self.duration:
            try:
                b.perf_buffer_poll(timeout=100)
            except KeyboardInterrupt:
                break

        elapsed = time.time() - start
        print(f"\n[Network Profiler] Stopped after {elapsed:.1f} seconds.")

        self._print_summary(b, elapsed)
        self._save_results()

    def _print_summary(self, b, elapsed):
        print("\n" + "=" * 70)
        print("NETWORK PROFILING SUMMARY")
        print("=" * 70)

        # TCP send stats
        key = ct.c_uint32(0)
        send_stats = b["tcp_send_stats"]
        if key in send_stats:
            s = send_stats[key]
            print(f"\nTCP Send:")
            print(f"  Count: {s.count}")
            if s.count > 0:
                print(f"  Avg latency: "
                      f"{s.total_latency_ns / s.count / 1000:.1f} us")
                print(f"  Min latency: {s.min_latency_ns / 1000:.1f} us")
                print(f"  Max latency: {s.max_latency_ns / 1000:.1f} us")

        # TCP recv stats
        recv_stats = b["tcp_recv_stats"]
        if key in recv_stats:
            r = recv_stats[key]
            print(f"\nTCP Recv:")
            print(f"  Count: {r.count}")
            if r.count > 0:
                print(f"  Avg latency: "
                      f"{r.total_latency_ns / r.count / 1000:.1f} us")
                print(f"  Min latency: {r.min_latency_ns / 1000:.1f} us")
                print(f"  Max latency: {r.max_latency_ns / 1000:.1f} us")

        # Net device xmit count
        xmit = b["net_xmit_count"]
        if key in xmit:
            print(f"\nNetwork Device Transmissions: {xmit[key].value}")
            print(f"  Rate: {xmit[key].value / elapsed:.0f} xmit/sec")

        # Per-process breakdown
        proc_events = defaultdict(lambda: defaultdict(int))
        for e in self.events:
            proc_events[e["comm"]][e["event_type"]] += 1

        if proc_events:
            print(f"\nPer-Process Network Activity:")
            print(f"  {'Process':<20} {'tcp_send':>10} {'tcp_recv':>10}")
            print(f"  {'-' * 40}")
            for comm, types in sorted(proc_events.items(),
                                       key=lambda x: -sum(x[1].values())):
                print(f"  {comm:<20} "
                      f"{types.get('tcp_send', 0):>10} "
                      f"{types.get('tcp_recv', 0):>10}")

    def _save_results(self):
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp_ns", "pid", "comm", "event_type", "latency_ns", "bytes"
            ])
            writer.writeheader()
            writer.writerows(self.events)

        print(f"\n[Network Profiler] Results saved to: {self.output_file}")
        print(f"  ({len(self.events)} events)")


def main():
    parser = argparse.ArgumentParser(
        description="eBPF Network Stack Profiler - Group 21"
    )
    parser.add_argument("--duration", type=int, default=60,
                        help="Profiling duration in seconds (default: 60)")
    parser.add_argument("--output", type=str,
                        default="21_net_results.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERROR: This script requires root privileges.")
        print("Run with: sudo python3 21_net_profiler.py")
        sys.exit(1)

    profiler = NetworkProfiler(
        duration=args.duration,
        output_file=args.output,
    )
    profiler.run()


if __name__ == "__main__":
    main()
