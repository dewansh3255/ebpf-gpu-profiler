#!/usr/bin/env python3
"""
perfetto_exporter.py
Merge the suite's eBPF + NVML telemetry into a single Chrome Trace Event Format
JSON that loads directly in https://ui.perfetto.dev .

All collectors timestamp on CLOCK_MONOTONIC nanoseconds (eBPF
`bpf_ktime_get_ns()` and the NVML monitor's `ts_mono_ns`), so the streams share
one time base. This exporter:

  * normalises every timestamp to microseconds relative to the earliest event,
  * renders CUDA driver-API calls (LAUNCH_QUEUE / HW_EXEC_SYNC / MEM_TRANSFER)
    as duration slices on their own tracks -- so you can literally see the GPU
    idle gaps between a sync completing and the next kernel launch,
  * renders NVML metrics (util, power, PCIe TX/RX, memory) as counter tracks,
  * renders eBPF TCP send/recv as duration slices.

Chrome Trace Event reference:
  ph "X" = complete (duration) slice with ts + dur
  ph "C" = counter sample (args -> series values)
  ph "M" = metadata (process_name / thread_name labels)

Usage:
    python3 perfetto_exporter.py \
        --nvml results/native/nvml_gpu.csv \
        --cuda results/native/cuda_trace.csv \
        --net  results/native/net_results.csv \
        --output results/native/perfetto_trace.json
"""

import argparse
import csv
import json
import os

# Synthetic process ids -> Perfetto "process" swim-lanes.
PID_GPU_COUNTERS = 10
PID_CUDA = 20
PID_NET = 30

# Stable thread ids for CUDA event tracks.
CUDA_TRACKS = {
    "LAUNCH_QUEUE": 1,   # CPU-side driver enqueue overhead
    "HW_EXEC_SYNC": 2,   # GPU execution + sync (CPU blocked)
    "MEM_TRANSFER": 3,   # PCIe host<->device copies
}


def _read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_trace(nvml_rows, cuda_rows, net_rows):
    events = []

    # ---- pass 1: find global min timestamp (ns) across all sources ----
    min_ns = None

    def consider(ns):
        nonlocal min_ns
        if ns > 0 and (min_ns is None or ns < min_ns):
            min_ns = ns

    for r in nvml_rows:
        consider(_to_int(r.get("ts_mono_ns")))
    for r in cuda_rows:
        end = _to_int(r.get("ts_mono_ns"))
        consider(end - _to_int(r.get("duration_ns")))
    for r in net_rows:
        consider(_to_int(r.get("timestamp_ns")))

    if min_ns is None:
        min_ns = 0

    def us(ns):
        return (ns - min_ns) / 1000.0

    # ---- process / thread labels ----
    def meta(pid, tid, name, key="thread_name"):
        events.append({"name": key, "ph": "M", "pid": pid, "tid": tid,
                       "args": {"name": name}})

    events.append({"name": "process_name", "ph": "M", "pid": PID_GPU_COUNTERS,
                   "tid": 0, "args": {"name": "GPU NVML Counters"}})
    events.append({"name": "process_name", "ph": "M", "pid": PID_CUDA,
                   "tid": 0, "args": {"name": "CUDA Driver API (eBPF uprobes)"}})
    events.append({"name": "process_name", "ph": "M", "pid": PID_NET,
                   "tid": 0, "args": {"name": "Network TCP (eBPF)"}})
    for name, tid in CUDA_TRACKS.items():
        meta(PID_CUDA, tid, name)

    # ---- CUDA slices ----
    for r in cuda_rows:
        etype = r.get("event_type", "")
        tid = CUDA_TRACKS.get(etype)
        if tid is None:
            continue
        end_ns = _to_int(r.get("ts_mono_ns"))
        dur_ns = _to_int(r.get("duration_ns"))
        begin_ns = end_ns - dur_ns
        events.append({
            "name": etype,
            "cat": "cuda",
            "ph": "X",
            "ts": us(begin_ns),
            "dur": dur_ns / 1000.0,
            "pid": PID_CUDA,
            "tid": tid,
            "args": {
                "duration_us": round(dur_ns / 1000.0, 3),
                "exec_gap_us": round(_to_int(r.get("exec_gap_ns")) / 1000.0, 3),
                "thread": r.get("tid"),
                "comm": r.get("comm"),
            },
        })

    # ---- NVML counters ----
    # Each metric becomes its own counter track per GPU so they stack nicely.
    metric_cols = [
        ("gpu_util_pct", "util_%"),
        ("power_w", "power_W"),
        ("pcie_tx_mibps", "pcie_tx_MiBps"),
        ("pcie_rx_mibps", "pcie_rx_MiBps"),
        ("mem_used_mib", "mem_used_MiB"),
    ]
    for r in nvml_rows:
        ts = us(_to_int(r.get("ts_mono_ns")))
        gpu = _to_int(r.get("gpu_index"))
        for col, series in metric_cols:
            val = _to_float(r.get(col), -1)
            if val < 0:
                continue
            events.append({
                "name": f"GPU{gpu} {series}",
                "ph": "C",
                "ts": ts,
                "pid": PID_GPU_COUNTERS,
                "tid": gpu,
                "args": {series: val},
            })

    # ---- network slices ----
    for r in net_rows:
        ts_ns = _to_int(r.get("timestamp_ns"))
        dur_ns = _to_int(r.get("latency_ns"))
        events.append({
            "name": r.get("event_type", "net"),
            "cat": "net",
            "ph": "X",
            "ts": us(ts_ns - dur_ns),
            "dur": dur_ns / 1000.0,
            "pid": PID_NET,
            "tid": 1,
            "args": {
                "bytes": _to_int(r.get("bytes")),
                "comm": r.get("comm"),
                "latency_us": round(dur_ns / 1000.0, 3),
            },
        })

    return {"traceEvents": events, "displayTimeUnit": "ms"}


def main():
    parser = argparse.ArgumentParser(
        description="Serialize eBPF + NVML telemetry into a Perfetto trace JSON")
    parser.add_argument("--nvml", type=str, default=None, help="NVML CSV")
    parser.add_argument("--cuda", type=str, default=None, help="CUDA uprobe CSV")
    parser.add_argument("--net", type=str, default=None, help="Network profiler CSV")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    nvml_rows = _read_csv(args.nvml)
    cuda_rows = _read_csv(args.cuda)
    net_rows = _read_csv(args.net)

    if not (nvml_rows or cuda_rows or net_rows):
        print("ERROR: no input data found. Pass at least one of --nvml/--cuda/--net.")
        raise SystemExit(1)

    trace = build_trace(nvml_rows, cuda_rows, net_rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(trace, f)

    n = len(trace["traceEvents"])
    print(f"[Perfetto] wrote {n:,} events -> {args.output}")
    print(f"[Perfetto] sources: nvml={len(nvml_rows)} cuda={len(cuda_rows)} "
          f"net={len(net_rows)} rows")
    print("[Perfetto] open it at https://ui.perfetto.dev (Open trace file).")


if __name__ == "__main__":
    main()
