#!/usr/bin/env python3
from bcc import BPF
import time
import csv
import socket
import struct

# ==========================================
# 1. eBPF C Code (Kernel Space)
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/sched.h>

// --- Data Structures ---
struct network_event_t {
    u64 ts_ns;
    u32 src_ip;
    u32 dst_ip;
    u32 payload_size;
    char direction[10];
};
BPF_PERF_OUTPUT(network_events);

struct drop_event_t {
    u64 ts_ns;
    u32 src_ip;
    u32 dst_ip;
    u32 drop_count;
};
BPF_PERF_OUTPUT(drop_events);

struct sched_event_t {
    u64 ts_ns;
    u32 pid;
    u64 delay_ns;
};
BPF_PERF_OUTPUT(sched_events);

// Hash maps to track timestamps
BPF_HASH(wakeup_time, u32, u64); // Tracks when a process was told to wake up
BPF_HASH(drop_counter, u32, u32); // Counts TCP retransmits

// --- 1. Network Probes (tcp_sendmsg / tcp_recvmsg) ---
int trace_tcp_sendmsg(struct pt_regs *ctx, struct sock *sk, struct msghdr *msg, size_t size) {
    u32 daddr = sk->sk_daddr;
    u32 saddr = sk->sk_rcv_saddr;

    struct network_event_t event = {};
    event.ts_ns = bpf_ktime_get_ns();
    event.src_ip = saddr;
    event.dst_ip = daddr;
    event.payload_size = size;
    __builtin_strcpy(event.direction, "OUT_SEND");

    network_events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}

int trace_tcp_recvmsg(struct pt_regs *ctx, struct sock *sk, struct msghdr *msg, size_t len) {
    u32 daddr = sk->sk_daddr;
    u32 saddr = sk->sk_rcv_saddr;

    struct network_event_t event = {};
    event.ts_ns = bpf_ktime_get_ns();
    event.src_ip = daddr; // Remote IP
    event.dst_ip = saddr; // Local IP
    event.payload_size = len;
    __builtin_strcpy(event.direction, "IN_RECV");

    network_events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}

// --- 2. sk_buff Drop Tracker (tcp_retransmit_skb) ---
int trace_tcp_retransmit(struct pt_regs *ctx, struct sock *sk) {
    u32 daddr = sk->sk_daddr;
    u32 saddr = sk->sk_rcv_saddr;
    
    struct drop_event_t event = {};
    event.ts_ns = bpf_ktime_get_ns();
    event.src_ip = saddr;
    event.dst_ip = daddr;
    event.drop_count = 1;

    drop_events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}

// --- 3. CPU Scheduler Tracker ---
TRACEPOINT_PROBE(sched, sched_wakeup) {
    u32 pid = args->pid;
    u64 ts = bpf_ktime_get_ns();
    wakeup_time.update(&pid, &ts);
    return 0;
}

TRACEPOINT_PROBE(sched, sched_switch) {
    u32 next_pid = args->next_pid;
    u64 *ts_ptr = wakeup_time.lookup(&next_pid);
    
    if (ts_ptr) {
        u64 delay = bpf_ktime_get_ns() - *ts_ptr;
        
        // Only send to Python if delay > 1ms (to filter noise)
        if (delay > 1000000) { 
            struct sched_event_t event = {};
            event.ts_ns = bpf_ktime_get_ns();
            event.pid = next_pid;
            event.delay_ns = delay;
            sched_events.perf_submit(args, &event, sizeof(event));
        }
        wakeup_time.delete(&next_pid);
    }
    return 0;
}
"""

# ==========================================
# 2. Python User Space
# ==========================================
def int_to_ip(ip_int):
    return socket.inet_ntoa(struct.pack("<L", ip_int))

if __name__ == "__main__":
    print("[+] Compiling eBPF Programs (Phase 1)...")
    
    # KERNEL 6.8+ FIX: Inject missing structs/macros directly into compiler
    bpf_cflags = [
        "-Dbpf_wq=bpf_timer",
        "-DBPF_LOAD_ACQ=1",
        "-DBPF_STORE_REL=2"
    ]
    
    b = BPF(text=bpf_text, cflags=bpf_cflags)

    # Attach Probes
    b.attach_kprobe(event="tcp_sendmsg", fn_name="trace_tcp_sendmsg")
    b.attach_kprobe(event="tcp_recvmsg", fn_name="trace_tcp_recvmsg")
    b.attach_kprobe(event="tcp_retransmit_skb", fn_name="trace_tcp_retransmit")
    
    csv_filename = f"network_trace_PHASE1_{int(time.time())}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time", "Event_Type", "Direction", "Src_IP", "Dest_IP", "Payload_MB_or_Delay_MS"])

        def print_network_event(cpu, data, size):
            event = b["network_events"].event(data)
            payload_mb = event.payload_size / (1024 * 1024)
            if payload_mb > 0.01:
                t = time.strftime('%H:%M:%S')
                src = int_to_ip(event.src_ip)
                dst = int_to_ip(event.dst_ip)
                direction = event.direction.decode('utf-8')
                writer.writerow([t, "NETWORK_IO", direction, src, dst, round(payload_mb, 4)])
                csvfile.flush()

        def print_drop_event(cpu, data, size):
            event = b["drop_events"].event(data)
            t = time.strftime('%H:%M:%S')
            src = int_to_ip(event.src_ip)
            dst = int_to_ip(event.dst_ip)
            writer.writerow([t, "TCP_RETRANSMIT", "DROP", src, dst, 1.0])
            print(f"[!] WI-FI DROP DETECTED! Retransmitting packet to {dst}")
            csvfile.flush()

        def print_sched_event(cpu, data, size):
            event = b["sched_events"].event(data)
            delay_ms = event.delay_ns / 1000000.0
            t = time.strftime('%H:%M:%S')
            writer.writerow([t, "SCHED_DELAY", "CPU_WAIT", "LOCAL", "LOCAL", round(delay_ms, 4)])
            csvfile.flush()

        b["network_events"].open_perf_buffer(print_network_event)
        b["drop_events"].open_perf_buffer(print_drop_event)
        b["sched_events"].open_perf_buffer(print_sched_event)

        print(f"[✓] Tracing Network, Drops, and CPU Scheduler. Saving to {csv_filename}")
        print("[!] Press Ctrl+C to stop.")

        try:
            while True:
                b.perf_buffer_poll()
        except KeyboardInterrupt:
            print("\n[+] Tracing stopped. Data saved.")