#!/usr/bin/env python3
"""
21_analyze_cpu.py
Group 21 - GRS Project Part A

Post-processing script for CPU profiler output. Reads the large
context switch and scheduling latency CSV files and produces
summary statistics (per-process breakdown, percentile distributions,
context switch rates over time).

Usage:
    python3 21_analyze_cpu.py --dir results/native
    python3 21_analyze_cpu.py --dir results/container
    python3 21_analyze_cpu.py --dir results/native --dir2 results/container

Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
import numpy as np


# ML-related process name patterns to filter for
ML_PROCESS_PATTERNS = [
    "python", "python3", "pt_main_thread", "pt_elastic",
    "pt_tcpstore", "torch", "nccl", "cuda", "gpu",
    "torchrun", "pt_autograd",
]


def is_ml_process(comm):
    """Check if a process name is related to the ML workload."""
    comm_lower = comm.lower()
    for pattern in ML_PROCESS_PATTERNS:
        if pattern in comm_lower:
            return True
    return False


def analyze_ctx_switches(filepath, max_rows=None):
    """Analyze context switch CSV file."""
    print(f"  Reading context switches from {filepath}...")

    total_rows = 0
    process_switches = defaultdict(int)      # comm -> count switched TO
    process_runtime_ns = defaultdict(int)     # comm -> total runtime
    timestamps = []
    ml_switches = 0
    total_switches = 0

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if max_rows and total_rows > max_rows:
                break

            next_comm = row.get("next_comm", "")
            prev_comm = row.get("prev_comm", "")

            process_switches[next_comm] += 1
            total_switches += 1

            runtime = int(row.get("prev_runtime_ns", 0))
            if runtime > 0:
                process_runtime_ns[prev_comm] += runtime

            ts = int(row.get("timestamp_ns", 0))
            if ts > 0:
                timestamps.append(ts)

            if is_ml_process(next_comm) or is_ml_process(prev_comm):
                ml_switches += 1

    # Calculate time window
    if timestamps:
        duration_s = (max(timestamps) - min(timestamps)) / 1e9
    else:
        duration_s = 1.0

    # Top processes
    top_by_switches = sorted(process_switches.items(), key=lambda x: -x[1])[:20]
    top_by_runtime = sorted(process_runtime_ns.items(), key=lambda x: -x[1])[:20]

    # Context switch rate over time (1-second bins)
    cs_rate_timeline = []
    if timestamps:
        start_ts = min(timestamps)
        for ts in timestamps:
            bin_sec = int((ts - start_ts) / 1e9)
            while len(cs_rate_timeline) <= bin_sec:
                cs_rate_timeline.append(0)
            cs_rate_timeline[bin_sec] += 1

    result = {
        "total_context_switches": total_switches,
        "total_rows_read": total_rows,
        "duration_seconds": round(duration_s, 2),
        "switches_per_second": round(total_switches / duration_s, 0) if duration_s > 0 else 0,
        "ml_related_switches": ml_switches,
        "ml_switch_pct": round(100.0 * ml_switches / total_switches, 2) if total_switches > 0 else 0,
        "top_processes_by_switches": [
            {"comm": comm, "count": count, "pct": round(100.0 * count / total_switches, 2)}
            for comm, count in top_by_switches
        ],
        "top_processes_by_runtime_ms": [
            {"comm": comm, "runtime_ms": round(runtime / 1e6, 2)}
            for comm, runtime in top_by_runtime
        ],
        "cs_rate_per_second": cs_rate_timeline,
    }
    return result


def analyze_sched_latency(filepath, max_rows=None):
    """Analyze scheduling latency CSV file."""
    print(f"  Reading scheduling latencies from {filepath}...")

    latencies_all = []
    latencies_ml = []
    per_process = defaultdict(list)
    total_rows = 0

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if max_rows and total_rows > max_rows:
                break

            latency_ns = int(row.get("latency_ns", 0))
            comm = row.get("comm", "")

            latencies_all.append(latency_ns)
            per_process[comm].append(latency_ns)

            if is_ml_process(comm):
                latencies_ml.append(latency_ns)

    def compute_stats(latencies):
        if not latencies:
            return {}
        arr = np.array(latencies, dtype=np.float64)
        return {
            "count": len(arr),
            "mean_us": round(float(np.mean(arr)) / 1000, 2),
            "median_us": round(float(np.median(arr)) / 1000, 2),
            "p50_us": round(float(np.percentile(arr, 50)) / 1000, 2),
            "p90_us": round(float(np.percentile(arr, 90)) / 1000, 2),
            "p95_us": round(float(np.percentile(arr, 95)) / 1000, 2),
            "p99_us": round(float(np.percentile(arr, 99)) / 1000, 2),
            "max_us": round(float(np.max(arr)) / 1000, 2),
            "min_us": round(float(np.min(arr)) / 1000, 2),
            "std_us": round(float(np.std(arr)) / 1000, 2),
        }

    # Per-process stats for top processes
    process_stats = {}
    top_procs = sorted(per_process.items(), key=lambda x: -len(x[1]))[:15]
    for comm, lats in top_procs:
        process_stats[comm] = compute_stats(lats)

    result = {
        "total_events": total_rows,
        "all_processes": compute_stats(latencies_all),
        "ml_processes_only": compute_stats(latencies_ml),
        "per_process_top15": process_stats,
    }
    return result


def analyze_directory(results_dir):
    """Analyze all CPU profiling data in a directory."""
    print(f"\nAnalyzing CPU data in: {results_dir}")

    results = {"directory": results_dir}

    # Context switches
    cs_file = os.path.join(results_dir, "21_cpu_results_ctx_switches.csv")
    if os.path.exists(cs_file):
        results["context_switches"] = analyze_ctx_switches(cs_file)
    else:
        print(f"  WARNING: {cs_file} not found")

    # Scheduling latency
    lat_file = os.path.join(results_dir, "21_cpu_results_sched_latency.csv")
    if os.path.exists(lat_file):
        results["scheduling_latency"] = analyze_sched_latency(lat_file)
    else:
        print(f"  WARNING: {lat_file} not found")

    return results


def print_comparison(native_results, container_results):
    """Print side-by-side comparison."""
    print("\n" + "=" * 80)
    print("CPU PROFILING COMPARISON: NATIVE vs CONTAINER")
    print("=" * 80)

    n_cs = native_results.get("context_switches", {})
    c_cs = container_results.get("context_switches", {})

    if n_cs and c_cs:
        print("\n--- Context Switches ---")
        print(f"  {'Metric':<35} {'Native':>15} {'Container':>15} {'Overhead':>12}")
        print(f"  {'-'*77}")

        metrics = [
            ("Total switches", "total_context_switches"),
            ("Switches/sec", "switches_per_second"),
            ("ML-related switches", "ml_related_switches"),
            ("ML switch %", "ml_switch_pct"),
        ]
        for label, key in metrics:
            nv = n_cs.get(key, 0)
            cv = c_cs.get(key, 0)
            overhead = f"{((cv - nv) / nv * 100):.1f}%" if nv > 0 else "N/A"
            print(f"  {label:<35} {nv:>15} {cv:>15} {overhead:>12}")

    n_lat = native_results.get("scheduling_latency", {})
    c_lat = container_results.get("scheduling_latency", {})

    if n_lat and c_lat:
        print("\n--- Scheduling Latency (All Processes) ---")
        n_all = n_lat.get("all_processes", {})
        c_all = c_lat.get("all_processes", {})

        print(f"  {'Metric':<35} {'Native':>15} {'Container':>15} {'Overhead':>12}")
        print(f"  {'-'*77}")

        for label, key in [("Mean (us)", "mean_us"), ("Median (us)", "median_us"),
                           ("P95 (us)", "p95_us"), ("P99 (us)", "p99_us"),
                           ("Max (us)", "max_us"), ("Std Dev (us)", "std_us")]:
            nv = n_all.get(key, 0)
            cv = c_all.get(key, 0)
            overhead = f"{((cv - nv) / nv * 100):.1f}%" if nv > 0 else "N/A"
            print(f"  {label:<35} {nv:>15.2f} {cv:>15.2f} {overhead:>12}")

        print("\n--- Scheduling Latency (ML Processes Only) ---")
        n_ml = n_lat.get("ml_processes_only", {})
        c_ml = c_lat.get("ml_processes_only", {})

        if n_ml and c_ml:
            print(f"  {'Metric':<35} {'Native':>15} {'Container':>15} {'Overhead':>12}")
            print(f"  {'-'*77}")
            for label, key in [("Count", "count"), ("Mean (us)", "mean_us"),
                               ("Median (us)", "median_us"), ("P95 (us)", "p95_us"),
                               ("P99 (us)", "p99_us")]:
                nv = n_ml.get(key, 0)
                cv = c_ml.get(key, 0)
                if isinstance(nv, int):
                    overhead = f"{((cv - nv) / nv * 100):.1f}%" if nv > 0 else "N/A"
                    print(f"  {label:<35} {nv:>15} {cv:>15} {overhead:>12}")
                else:
                    overhead = f"{((cv - nv) / nv * 100):.1f}%" if nv > 0 else "N/A"
                    print(f"  {label:<35} {nv:>15.2f} {cv:>15.2f} {overhead:>12}")


def main():
    parser = argparse.ArgumentParser(
        description="CPU Profiling Data Analyzer - Group 21"
    )
    parser.add_argument("--dir", type=str, default="results/native",
                        help="Results directory to analyze")
    parser.add_argument("--dir2", type=str, default=None,
                        help="Second directory for comparison (e.g., results/container)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for summary stats")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows to read (for faster testing)")
    args = parser.parse_args()

    results1 = analyze_directory(args.dir)

    # Print single-directory summary
    cs = results1.get("context_switches", {})
    if cs:
        print(f"\n  Total context switches: {cs.get('total_context_switches', 0):,}")
        print(f"  Duration: {cs.get('duration_seconds', 0)}s")
        print(f"  Rate: {cs.get('switches_per_second', 0):,.0f}/sec")
        print(f"  ML-related: {cs.get('ml_related_switches', 0):,} ({cs.get('ml_switch_pct', 0)}%)")

    lat = results1.get("scheduling_latency", {})
    if lat:
        all_stats = lat.get("all_processes", {})
        print(f"\n  Scheduling latency (all):")
        print(f"    Mean: {all_stats.get('mean_us', 0):.2f} us")
        print(f"    P50:  {all_stats.get('p50_us', 0):.2f} us")
        print(f"    P95:  {all_stats.get('p95_us', 0):.2f} us")
        print(f"    P99:  {all_stats.get('p99_us', 0):.2f} us")

    # Comparison mode
    if args.dir2:
        results2 = analyze_directory(args.dir2)
        print_comparison(results1, results2)

        if args.output:
            combined = {"native": results1, "container": results2}
            # Remove large timeline data for JSON output
            for key in ["native", "container"]:
                if "context_switches" in combined[key]:
                    combined[key]["context_switches"].pop("cs_rate_per_second", None)
            with open(args.output, "w") as f:
                json.dump(combined, f, indent=2, default=str)
            print(f"\nSummary saved to: {args.output}")
    elif args.output:
        if "context_switches" in results1:
            results1["context_switches"].pop("cs_rate_per_second", None)
        with open(args.output, "w") as f:
            json.dump(results1, f, indent=2, default=str)
        print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    main()
