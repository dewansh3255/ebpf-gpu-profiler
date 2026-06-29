#!/usr/bin/env python3
"""
gpu_idle_gaps.py
Compute and visualise "GPU Idle Gaps" -- the CPU-starvation metric.

Definition: a GPU idle gap is the wall-clock interval between a synchronisation
returning (i.e. the device finished its queued work and the CPU unblocked) and
the *next* kernel launch being issued on the same thread. During that window the
device has nothing to run because the CPU has not yet handed it the next kernel.
Large/frequent gaps == the workload is kernel-launch-bound and the CPU (or the
container's scheduling/syscall overhead) is starving the GPU.

Input: the CSV produced by src/bpf_kernel/cuda_uprobe_monitor.py with columns
    ts_mono_ns, event_type, duration_ns, exec_gap_ns, pid, tid, comm
where ts_mono_ns is the call-completion timestamp (CLOCK_MONOTONIC) and a launch
begins at (ts_mono_ns - duration_ns).

Per thread we walk events in time order: after each HW_EXEC_SYNC completion we
measure the delay until the next LAUNCH_QUEUE begin.

Usage:
    # single run
    python3 gpu_idle_gaps.py --cuda results/native/cuda_trace.csv

    # native vs container comparison (+ plot)
    python3 gpu_idle_gaps.py \
        --cuda results/native/cuda_trace.csv \
        --cuda2 results/container/cuda_trace.csv \
        --label1 Native --label2 Container \
        --plot results/plots/gpu_idle_gaps.png \
        --output results/gpu_idle_gaps.json
"""

import argparse
import csv
import json
import os

import numpy as np


def load_events(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "ts": int(float(r["ts_mono_ns"])),
                    "etype": r["event_type"],
                    "dur": int(float(r["duration_ns"])),
                    "tid": int(float(r["tid"])),
                })
            except (KeyError, ValueError):
                continue
    return rows


def compute_idle_gaps(rows):
    """Return list of idle-gap durations in nanoseconds."""
    by_tid = {}
    for e in rows:
        by_tid.setdefault(e["tid"], []).append(e)

    gaps = []
    for tid, evs in by_tid.items():
        evs.sort(key=lambda e: e["ts"])
        last_sync_end = None
        for e in evs:
            if e["etype"] == "HW_EXEC_SYNC":
                last_sync_end = e["ts"]  # completion time
            elif e["etype"] == "LAUNCH_QUEUE":
                if last_sync_end is not None:
                    launch_begin = e["ts"] - e["dur"]
                    gap = launch_begin - last_sync_end
                    if gap > 0:
                        gaps.append(gap)
                    last_sync_end = None  # only first launch after a sync
    return gaps


def summarise(gaps, total_window_ns=None):
    if not gaps:
        return {"count": 0}
    arr = np.array(gaps, dtype=np.float64)
    s = {
        "count": int(arr.size),
        "total_idle_ms": round(float(arr.sum()) / 1e6, 3),
        "mean_us": round(float(arr.mean()) / 1e3, 3),
        "median_us": round(float(np.median(arr)) / 1e3, 3),
        "p95_us": round(float(np.percentile(arr, 95)) / 1e3, 3),
        "p99_us": round(float(np.percentile(arr, 99)) / 1e3, 3),
        "max_us": round(float(arr.max()) / 1e3, 3),
    }
    if total_window_ns:
        s["idle_fraction_pct"] = round(100.0 * arr.sum() / total_window_ns, 2)
    return s


def window_ns(rows):
    ts = [r["ts"] for r in rows]
    return (max(ts) - min(ts)) if ts else None


def analyse(path, label):
    rows = load_events(path)
    gaps = compute_idle_gaps(rows)
    summary = summarise(gaps, window_ns(rows))
    print(f"\n[{label}] {path}")
    print(f"  events={len(rows)}  idle_gaps={summary.get('count', 0)}")
    if summary.get("count", 0):
        print(f"  total idle  : {summary['total_idle_ms']} ms"
              + (f"  ({summary['idle_fraction_pct']}% of window)"
                 if "idle_fraction_pct" in summary else ""))
        print(f"  mean/median : {summary['mean_us']} / {summary['median_us']} us")
        print(f"  p95/p99/max : {summary['p95_us']} / {summary['p99_us']} "
              f"/ {summary['max_us']} us")
    return gaps, summary


def plot_comparison(g1, l1, g2, l2, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    NC, CC = "#2196F3", "#FF5722"

    # (a) CDF
    for gaps, label, color in [(g1, l1, NC), (g2, l2, CC)]:
        if not gaps:
            continue
        arr = np.sort(np.array(gaps, dtype=np.float64) / 1e3)  # us
        p99 = np.percentile(arr, 99)
        clipped = arr[arr <= p99]
        cdf = np.arange(1, len(clipped) + 1) / len(arr)
        axes[0].plot(clipped, cdf, label=f"{label} (n={len(arr):,})",
                     color=color, linewidth=2)
    axes[0].set_xlabel("GPU idle gap (us)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("(a) GPU Idle Gap CDF (<=p99)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # (b) summary bars
    s1, s2 = summarise(g1), summarise(g2)
    metrics = [("mean_us", "Mean"), ("median_us", "Median"),
               ("p95_us", "P95"), ("p99_us", "P99")]
    x = np.arange(len(metrics))
    w = 0.35
    axes[1].bar(x - w / 2, [s1.get(k, 0) for k, _ in metrics], w,
                label=l1, color=NC)
    axes[1].bar(x + w / 2, [s2.get(k, 0) for k, _ in metrics], w,
                label=l2, color=CC)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([lbl for _, lbl in metrics])
    axes[1].set_ylabel("GPU idle gap (us)")
    axes[1].set_title("(b) GPU Idle Gap Percentiles")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("GPU Idle Gaps (CPU Starvation): "
                 f"{l1} vs {l2}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GPU idle-gap (CPU starvation) analysis")
    parser.add_argument("--cuda", required=True, help="primary cuda_trace.csv")
    parser.add_argument("--cuda2", default=None, help="second cuda_trace.csv for comparison")
    parser.add_argument("--label1", default="Run1")
    parser.add_argument("--label2", default="Run2")
    parser.add_argument("--plot", default=None, help="output PNG (needs --cuda2)")
    parser.add_argument("--output", default=None, help="output JSON summary")
    args = parser.parse_args()

    g1, s1 = analyse(args.cuda, args.label1)
    result = {args.label1: s1}

    if args.cuda2:
        g2, s2 = analyse(args.cuda2, args.label2)
        result[args.label2] = s2
        if args.plot:
            plot_comparison(g1, args.label1, g2, args.label2, args.plot)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[JSON] saved -> {args.output}")


if __name__ == "__main__":
    main()
