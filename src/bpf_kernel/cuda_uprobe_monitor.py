#!/usr/bin/env python3
from bcc import BPF
import time
import csv

# ==========================================
# 1. eBPF C Code (Kernel & User Space Probes)
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct gpu_event_t {
    u64 ts_ns;
    u64 duration_ns;
    u32 pid;
    char event_type[20];
};
BPF_PERF_OUTPUT(gpu_events);

// Hash maps to track the start time
BPF_HASH(mem_start, u32, u64); 
BPF_HASH(compute_start, u32, u64);

// --- 1. Memory Transfers (Async PCIe Bus) ---
int trace_mem_entry(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 ts = bpf_ktime_get_ns();
    mem_start.update(&pid, &ts);
    return 0;
}
int trace_mem_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 *tsp = mem_start.lookup(&pid);
    if (tsp != 0) {
        u64 delta = bpf_ktime_get_ns() - *tsp;
        struct gpu_event_t event = {};
        event.ts_ns = bpf_ktime_get_ns();
        event.duration_ns = delta;
        event.pid = pid;
        __builtin_strcpy(event.event_type, "MEM_TRANSFER");
        gpu_events.perf_submit(ctx, &event, sizeof(event));
        mem_start.delete(&pid);
    }
    return 0;
}

// --- 2. Compute: Math Execution (Kernel Launch) ---
int trace_cuLaunchKernel_entry(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 ts = bpf_ktime_get_ns();
    compute_start.update(&pid, &ts);
    return 0;
}
int trace_cuLaunchKernel_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 *tsp = compute_start.lookup(&pid);
    if (tsp != 0) {
        u64 delta = bpf_ktime_get_ns() - *tsp;
        struct gpu_event_t event = {};
        event.ts_ns = bpf_ktime_get_ns();
        event.duration_ns = delta;
        event.pid = pid;
        __builtin_strcpy(event.event_type, "COMPUTE_MATH");
        gpu_events.perf_submit(ctx, &event, sizeof(event));
        compute_start.delete(&pid);
    }
    return 0;
}
"""

# ==========================================
# 2. Python User Space
# ==========================================
if __name__ == "__main__":
    print("[+] Compiling eGPU Hardware Probes (Phase 2)...")
    b = BPF(text=bpf_text)

    try:
        # Hook into Math operations (We know this works!)
        b.attach_uprobe(name="cuda", sym="cuLaunchKernel", fn_name="trace_cuLaunchKernel_entry")
        b.attach_uretprobe(name="cuda", sym="cuLaunchKernel", fn_name="trace_cuLaunchKernel_return")
        
        # List of internal Async/Sync Memory functions PyTorch might use via the Driver API
        mem_symbols = [
            "cuMemcpyHtoDAsync_v2",
            "cuMemcpyDtoHAsync_v2",
            "cuMemcpyAsync",
            "cuMemcpyHtoD_v2",
            "cuMemcpyDtoH_v2"
        ]
        
        hooked_count = 0
        for sym in mem_symbols:
            try:
                b.attach_uprobe(name="cuda", sym=sym, fn_name="trace_mem_entry")
                b.attach_uretprobe(name="cuda", sym=sym, fn_name="trace_mem_return")
                hooked_count += 1
            except Exception:
                pass # Symbol not found, ignore and try the next one
                
        if hooked_count > 0:
            print(f"[✓] Successfully attached to NVIDIA CUDA Driver API ({hooked_count} memory hooks active).")
        else:
            print("[!] Warning: Hooked Math, but could not find memory symbols in libcuda.")

    except Exception as e:
        print(f"[!] Fatal Error: Could not attach to libcuda.so. Error: {e}")
        exit(1)

    csv_filename = f"gpu_trace_PHASE2_{int(time.time())}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time", "Event_Type", "Duration_ms"])

        def print_gpu_event(cpu, data, size):
            event = b["gpu_events"].event(data)
            duration_ms = event.duration_ns / 1000000.0
            
            # Log all events (cuLaunchKernel is sub-ms but real)
            if duration_ms >= 0:
                t = time.strftime('%H:%M:%S')
                event_type = event.event_type.decode('utf-8')
                writer.writerow([t, event_type, round(duration_ms, 4)])
                csvfile.flush()
                print(f"  [{t}] {event_type:15s} {duration_ms:.4f} ms")

        b["gpu_events"].open_perf_buffer(print_gpu_event)

        print(f"[✓] Tracing GPU Math and PCIe Memory Transfers. Saving to {csv_filename}")
        print("[!] Press Ctrl+C to stop.")

        try:
            while True:
                b.perf_buffer_poll()
        except KeyboardInterrupt:
            print("\n[+] Tracing stopped. Data saved.")