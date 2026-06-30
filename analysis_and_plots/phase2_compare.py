#!/usr/bin/env python3
"""
phase2_compare.py
Bare-Metal vs Docker comparison for the Phase 2 (horizontal / federated)
runs on the RTX 3060 client.

Consumes the standard per-run output directory produced by
run_native_network.sh / run_docker_network.sh:

    results/phase2/<native|docker>_<arch>/
        net.csv            ts, pid, comm, event_type, latency_ns, bytes
        syscall.csv        syscall_nr, name, count, total_latency_ns, ...
        llc.csv            pid, comm, cgroup_id, containerized, llc_misses_est, ...
        cpu_sched_latency.csv  timestamp_ns, pid, comm, latency_ns
        cuda_trace.csv     ts_mono_ns, event_type, duration_ns, exec_gap_ns, ...
        nvml_gpu.csv       ts_mono_ns, ..., gpu_util_pct, power_w, pcie_tx_mibps, ...

For each (mode, arch) it extracts headline metrics and renders a polished
multi-panel comparison plus a JSON/CSV summary.

Usage:
    python3 phase2_compare.py --results-base results/phase2 \
        --archs resnet18 resnet50 --out results/phase2/plots
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NATIVE_COLOR = "#2E7D32"   # green  = bare metal
DOCKER_COLOR = "#1565C0"   # blue   = docker
csv.field_size_limit(10 ** 9)


# ----------------------------- loaders -----------------------------
def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def net_summary(run_dir):
    rows = _rows(os.path.join(run_dir, "net.csv"))
    if not rows:
        return {}
    lat = np.array([_f(r.get("latency_ns")) for r in rows], dtype=np.float64)
    by = np.array([_f(r.get("bytes")) for r in rows], dtype=np.float64)
    lat = lat[lat > 0]
    return {
        "net_events": len(rows),
        "net_total_mb": round(by.sum() / 1e6, 2),
        "net_lat_mean_us": round(float(lat.mean()) / 1e3, 2) if lat.size else 0,
        "net_lat_p99_us": round(float(np.percentile(lat, 99)) / 1e3, 2) if lat.size else 0,
    }


def syscall_summary(run_dir):
    rows = _rows(os.path.join(run_dir, "syscall.csv"))
    if not rows:
        return {}
    total = sum(int(_f(r.get("count"))) for r in rows)
    total_time = sum(_f(r.get("total_latency_ns")) for r in rows)
    return {
        "syscall_total": total,
        "syscall_time_ms": round(total_time / 1e6, 1),
        "syscall_distinct": len(rows),
    }


def llc_summary(run_dir):
    rows = _rows(os.path.join(run_dir, "llc.csv"))
    if not rows:
        return {}
    misses = sum(_f(r.get("llc_misses_est")) for r in rows)
    refs = sum(_f(r.get("llc_refs_est")) for r in rows)
    rate = (100.0 * misses / refs) if refs > 0 else 0.0
    return {
        "llc_misses_M": round(misses / 1e6, 2),
        "llc_miss_rate_pct": round(rate, 2),
    }


def sched_summary(run_dir):
    rows = _rows(os.path.join(run_dir, "cpu_sched_latency.csv"))
    if not rows:
        return {}
    lat = np.array([_f(r.get("latency_ns")) for r in rows], dtype=np.float64)
    lat = lat[lat > 0]
    if not lat.size:
        return {}
    return {
        "sched_lat_p50_us": round(float(np.percentile(lat, 50)) / 1e3, 2),
        "sched_lat_p99_us": round(float(np.percentile(lat, 99)) / 1e3, 2),
    }


def cuda_summary(run_dir):
    rows = _rows(os.path.join(run_dir, "cuda_trace.csv"))
    if not rows:
        return {}
    launch, sync = [], []
    by_tid = {}
    for r in rows:
        et = r.get("event_type")
        d = _f(r.get("duration_ns"))
        if et == "LAUNCH_QUEUE":
            launch.append(d)
        elif et == "HW_EXEC_SYNC":
            sync.append(d)
        try:
            tid = int(float(r["tid"]))
            by_tid.setdefault(tid, []).append(
                (int(float(r["ts_mono_ns"])), et, int(d)))
        except (KeyError, ValueError):
            pass

    # GPU idle gaps: sync-return -> next launch-begin, per thread
    gaps = []
    for tid, evs in by_tid.items():
        evs.sort(key=lambda x: x[0])
        last_sync_end = None
        for ts, et, d in evs:
            if et == "HW_EXEC_SYNC":
                last_sync_end = ts
            elif et == "LAUNCH_QUEUE" and last_sync_end is not None:
                gap = (ts - d) - last_sync_end
                if gap > 0:
                    gaps.append(gap)
                last_sync_end = None

    out = {}
    if launch:
        a = np.array(launch, dtype=np.float64)
        out["launch_queue_mean_us"] = round(float(a.mean()) / 1e3, 2)
        out["launch_queue_count"] = len(launch)
    if sync:
        a = np.array(sync, dtype=np.float64)
        out["hw_exec_sync_mean_us"] = round(float(a.mean()) / 1e3, 2)
    if gaps:
        a = np.array(gaps, dtype=np.float64)
        out["gpu_idle_gap_mean_us"] = round(float(a.mean()) / 1e3, 2)
        out["gpu_idle_total_ms"] = round(float(a.sum()) / 1e6, 1)
        out["gpu_idle_gap_count"] = len(gaps)
    out["_idle_gaps_us"] = (np.array(gaps) / 1e3) if gaps else np.array([])
    return out


def nvml_summary(run_dir):
    rows = _rows(os.path.join(run_dir, "nvml_gpu.csv"))
    if not rows:
        return {}
    def col(name):
        return np.array([_f(r.get(name), -1) for r in rows], dtype=np.float64)
    util = col("gpu_util_pct"); util = util[util >= 0]
    power = col("power_w"); power = power[power >= 0]
    tx = col("pcie_tx_mibps"); tx = tx[tx >= 0]
    rx = col("pcie_rx_mibps"); rx = rx[rx >= 0]
    return {
        "gpu_util_avg_pct": round(float(util.mean()), 1) if util.size else 0,
        "power_avg_w": round(float(power.mean()), 1) if power.size else 0,
        "pcie_tx_avg_mibps": round(float(tx.mean()), 1) if tx.size else 0,
        "pcie_rx_avg_mibps": round(float(rx.mean()), 1) if rx.size else 0,
    }


def collect(run_dir):
    s = {}
    for fn in (net_summary, syscall_summary, llc_summary, sched_summary,
               cuda_summary, nvml_summary):
        try:
            s.update(fn(run_dir))
        except Exception as e:
            print(f"  [warn] {fn.__name__}({run_dir}): {e}")
    return s


# ----------------------------- plotting -----------------------------
def _grouped_bar(ax, archs, native_vals, docker_vals, title, ylabel):
    x = np.arange(len(archs))
    w = 0.36
    ax.bar(x - w / 2, native_vals, w, label="Bare-metal", color=NATIVE_COLOR)
    ax.bar(x + w / 2, docker_vals, w, label="Docker", color=DOCKER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, (n, d) in enumerate(zip(native_vals, docker_vals)):
        ax.text(i - w / 2, n, f"{n:g}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, d, f"{d:g}", ha="center", va="bottom", fontsize=7)


def make_overview(data, archs, out_path):
    """data[(mode, arch)] = summary dict."""
    panels = [
        ("net_total_mb", "Network bytes (host TCP)", "MB"),
        ("net_lat_p99_us", "Network p99 latency", "us"),
        ("syscall_total", "Total syscalls", "count"),
        ("llc_miss_rate_pct", "LLC miss rate", "%"),
        ("sched_lat_p99_us", "Scheduler p99 latency", "us"),
        ("gpu_idle_total_ms", "GPU idle (CPU starvation)", "ms"),
        ("launch_queue_mean_us", "CUDA launch-queue mean", "us"),
        ("hw_exec_sync_mean_us", "HW exec+sync mean", "us"),
        ("gpu_util_avg_pct", "GPU utilisation", "%"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        nat = [data.get(("native", a), {}).get(key, 0) for a in archs]
        doc = [data.get(("docker", a), {}).get(key, 0) for a in archs]
        _grouped_bar(ax, archs, nat, doc, title, ylabel)
    axes.flat[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Phase 2 (RTX 3060 Federated Client): Bare-Metal vs Docker",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def make_idle_cdf(data, archs, out_path):
    fig, axes = plt.subplots(1, len(archs), figsize=(7 * len(archs), 5), squeeze=False)
    for j, arch in enumerate(archs):
        ax = axes[0][j]
        for mode, color in (("native", NATIVE_COLOR), ("docker", DOCKER_COLOR)):
            g = data.get((mode, arch), {}).get("_idle_gaps_us")
            if g is None or len(g) == 0:
                continue
            arr = np.sort(g)
            p99 = np.percentile(arr, 99)
            clip = arr[arr <= p99]
            cdf = np.arange(1, len(clip) + 1) / len(arr)
            ax.plot(clip, cdf, label=f"{mode} (n={len(arr):,})",
                    color=color, linewidth=2)
        ax.set_title(f"GPU idle-gap CDF — {arch}", fontweight="bold")
        ax.set_xlabel("idle gap (us, <=p99)")
        ax.set_ylabel("CDF")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("GPU Idle Gaps (CPU starvation): Bare-Metal vs Docker",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Phase 2 bare-metal vs docker comparison")
    ap.add_argument("--results-base", default="results/phase2")
    ap.add_argument("--archs", nargs="+", default=["resnet18", "resnet50"])
    ap.add_argument("--out", default="results/phase2/plots")
    args = ap.parse_args()

    data, summary = {}, {}
    for arch in args.archs:
        for mode in ("native", "docker"):
            run_dir = os.path.join(args.results_base, f"{mode}_{arch}")
            if not os.path.isdir(run_dir):
                print(f"[skip] missing {run_dir}")
                continue
            print(f"[load] {run_dir}")
            s = collect(run_dir)
            data[(mode, arch)] = s
            summary[f"{mode}_{arch}"] = {k: v for k, v in s.items()
                                         if not k.startswith("_")}

    if not data:
        print("No run directories found. Nothing to do.")
        raise SystemExit(1)

    os.makedirs(args.out, exist_ok=True)
    make_overview(data, args.archs, os.path.join(args.out, "phase2_overview.png"))
    make_idle_cdf(data, args.archs, os.path.join(args.out, "phase2_gpu_idle_cdf.png"))

    sj = os.path.join(args.out, "phase2_summary.json")
    with open(sj, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[json] {sj}")

    # Console table
    print("\n" + "=" * 78)
    print("PHASE 2 SUMMARY (Bare-Metal vs Docker)")
    print("=" * 78)
    for key in sorted(summary):
        print(f"\n[{key}]")
        for k, v in summary[key].items():
            print(f"  {k:<24} {v}")


if __name__ == "__main__":
    main()
