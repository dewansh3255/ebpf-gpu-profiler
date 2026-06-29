#!/usr/bin/env python3
"""
21_compare_results.py
Group 21 - GRS Project Part A

Comprehensive comparative analysis and publication-quality visualization.
Reads native and container results, computes per-subsystem overhead,
and generates all comparison plots for the final report.

This script implements Objective 3: Cross-Scenario Comparison and
Overhead Attribution.

Usage:
    python3 21_compare_results.py
    python3 21_compare_results.py --native results/native --container results/container
    python3 21_compare_results.py --output-dir results/plots

Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---- Plot styling ----
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

NATIVE_COLOR = "#2196F3"   # Blue
CONTAINER_COLOR = "#FF5722" # Orange-Red
ACCENT_1 = "#4CAF50"        # Green
ACCENT_2 = "#9C27B0"        # Purple


# ============================================================
# DATA LOADING
# ============================================================

def load_training_json(filepath):
    """Load training results JSON."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def load_csv(filepath, max_rows=None):
    """Load CSV file into list of dicts."""
    if not os.path.exists(filepath):
        return []
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


def load_syscall_csv(filepath):
    """Load syscall results CSV."""
    if not os.path.exists(filepath):
        return []
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["count"] = int(row.get("count", 0))
                row["avg_latency_ns"] = float(row.get("avg_latency_ns", 0))
                row["min_latency_ns"] = float(row.get("min_latency_ns", 0))
                row["max_latency_ns"] = float(row.get("max_latency_ns", 0))
            except (ValueError, TypeError):
                continue
            rows.append(row)
    return rows


def load_gpu_csv(filepath):
    """Load GPU monitoring CSV."""
    if not os.path.exists(filepath):
        return []
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    if "." in str(v):
                        parsed[k] = float(v)
                    else:
                        parsed[k] = int(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def load_net_csv(filepath, max_rows=500000):
    """Load network profiling CSV."""
    if not os.path.exists(filepath):
        return []
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                row["latency_ns"] = int(row.get("latency_ns", 0))
                row["bytes"] = int(row.get("bytes", 0))
            except (ValueError, TypeError):
                row["latency_ns"] = 0
                row["bytes"] = 0
            rows.append(row)
    return rows


def load_sched_latency_csv(filepath, max_rows=2000000):
    """Load scheduling latency CSV (can be very large)."""
    if not os.path.exists(filepath):
        return []
    latencies = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                latencies.append(int(row.get("latency_ns", 0)))
            except (ValueError, TypeError):
                pass
    return latencies


# ============================================================
# PLOT GENERATORS
# ============================================================

def plot_training_comparison(native_json, container_json, output_dir):
    """Plot 1: Training time and throughput comparison."""
    if not native_json or not container_json:
        print("  Skipping training comparison (missing data)")
        return

    n_epochs = native_json.get("epochs", [])
    c_epochs = container_json.get("epochs", [])

    if not n_epochs or not c_epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Epoch training time
    n_times = [e["epoch_time_sec"] for e in n_epochs]
    c_times = [e["epoch_time_sec"] for e in c_epochs]
    epochs = range(1, len(n_times) + 1)

    x = np.arange(len(epochs))
    width = 0.35
    axes[0].bar(x - width/2, n_times, width, label="Native", color=NATIVE_COLOR, edgecolor="black", linewidth=0.5)
    axes[0].bar(x + width/2, c_times[:len(epochs)], width, label="Container", color=CONTAINER_COLOR, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Time (seconds)")
    axes[0].set_title("(a) Per-Epoch Training Time")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(e) for e in epochs])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # (b) Throughput
    n_throughput = [e["samples_per_sec"] for e in n_epochs]
    c_throughput = [e["samples_per_sec"] for e in c_epochs]

    axes[1].bar(x - width/2, n_throughput, width, label="Native", color=NATIVE_COLOR, edgecolor="black", linewidth=0.5)
    axes[1].bar(x + width/2, c_throughput[:len(epochs)], width, label="Container", color=CONTAINER_COLOR, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Throughput (samples/sec)")
    axes[1].set_title("(b) Training Throughput")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(e) for e in epochs])
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    # (c) Test accuracy progression
    n_acc = [e["test_accuracy"] for e in n_epochs]
    c_acc = [e["test_accuracy"] for e in c_epochs]

    axes[2].plot(list(epochs), n_acc, "o-", label="Native", color=NATIVE_COLOR, linewidth=2, markersize=6)
    axes[2].plot(list(range(1, len(c_acc)+1)), c_acc, "s-", label="Container", color=CONTAINER_COLOR, linewidth=2, markersize=6)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_title("(c) Model Convergence")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Training Performance: Native vs Containerized (ResNet-18 on CIFAR-10, DDP)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_training.png"))
    plt.close()
    print("  Saved compare_training.png")


def plot_total_time_comparison(native_json, container_json, output_dir):
    """Plot 2: Total training time bar chart with overhead annotation."""
    if not native_json or not container_json:
        return

    n_total = native_json.get("total_time_sec", 0)
    c_total = container_json.get("total_time_sec", 0)
    overhead_pct = ((c_total - n_total) / n_total * 100) if n_total > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Native\n(Bare-Metal)", "Container\n(Docker Bridge)"],
                  [n_total, c_total],
                  color=[NATIVE_COLOR, CONTAINER_COLOR],
                  edgecolor="black", linewidth=0.8, width=0.5)

    # Add value labels
    for bar, val in zip(bars, [n_total, c_total]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontweight="bold", fontsize=13)

    # Overhead annotation
    arrow_label = f"+{overhead_pct:.1f}% overhead" if overhead_pct > 0 else f"{overhead_pct:.1f}%"
    color = "red" if overhead_pct > 0 else "green"
    ax.annotate(arrow_label,
                xy=(1, c_total), xytext=(1.3, (n_total + c_total)/2),
                fontsize=13, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
                ha="left", va="center")

    ax.set_ylabel("Total Training Time (seconds)")
    ax.set_title("Total Training Time: Native vs Containerized", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_total_time.png"))
    plt.close()
    print("  Saved compare_total_time.png")


def plot_syscall_comparison(native_syscalls, container_syscalls, output_dir):
    """Plot 3: Syscall frequency comparison (grouped bar)."""
    if not native_syscalls or not container_syscalls:
        print("  Skipping syscall comparison (missing data)")
        return

    # Build lookup
    n_lookup = {r["name"]: r for r in native_syscalls}
    c_lookup = {r["name"]: r for r in container_syscalls}

    # Find top syscalls by combined count
    all_syscalls = set(n_lookup.keys()) | set(c_lookup.keys())
    combined = []
    for name in all_syscalls:
        n_count = n_lookup.get(name, {}).get("count", 0)
        c_count = c_lookup.get(name, {}).get("count", 0)
        combined.append((name, n_count, c_count))

    combined.sort(key=lambda x: -(x[1] + x[2]))
    top15 = combined[:15]

    names = [t[0] for t in top15]
    n_counts = [t[1] for t in top15]
    c_counts = [t[2] for t in top15]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # (a) Count comparison
    y = np.arange(len(names))
    height = 0.35
    axes[0].barh(y - height/2, n_counts, height, label="Native", color=NATIVE_COLOR, edgecolor="black", linewidth=0.3)
    axes[0].barh(y + height/2, c_counts, height, label="Container", color=CONTAINER_COLOR, edgecolor="black", linewidth=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("Total Count")
    axes[0].set_title("(a) System Call Frequency")
    axes[0].legend()
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else f"{x:.0f}"))
    axes[0].grid(axis="x", alpha=0.3)

    # (b) Latency comparison
    n_latencies = []
    c_latencies = []
    lat_names = []
    for name, _, _ in top15:
        n_lat = n_lookup.get(name, {}).get("avg_latency_ns", 0) / 1000  # to us
        c_lat = c_lookup.get(name, {}).get("avg_latency_ns", 0) / 1000
        if n_lat > 0 or c_lat > 0:
            lat_names.append(name)
            n_latencies.append(n_lat)
            c_latencies.append(c_lat)

    # Filter to reasonable latencies for visualization (< 100ms)
    filtered = [(n, nl, cl) for n, nl, cl in zip(lat_names, n_latencies, c_latencies)
                if nl < 100000 and cl < 100000]
    if filtered:
        lat_names, n_latencies, c_latencies = zip(*filtered)
        y2 = np.arange(len(lat_names))
        axes[1].barh(y2 - height/2, n_latencies, height, label="Native", color=NATIVE_COLOR, edgecolor="black", linewidth=0.3)
        axes[1].barh(y2 + height/2, c_latencies, height, label="Container", color=CONTAINER_COLOR, edgecolor="black", linewidth=0.3)
        axes[1].set_yticks(y2)
        axes[1].set_yticklabels(lat_names)
        axes[1].set_xlabel("Average Latency (microseconds)")
        axes[1].set_title("(b) System Call Average Latency")
        axes[1].legend()
        axes[1].invert_yaxis()
        axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("System Call Comparison: Native vs Containerized",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_syscalls.png"))
    plt.close()
    print("  Saved compare_syscalls.png")


def plot_syscall_overhead_table(native_syscalls, container_syscalls, output_dir):
    """Plot 4: Syscall overhead percentage table as figure."""
    if not native_syscalls or not container_syscalls:
        return

    n_lookup = {r["name"]: r for r in native_syscalls}
    c_lookup = {r["name"]: r for r in container_syscalls}

    # Key syscalls for ML workloads
    key_syscalls = ["futex", "read", "write", "ioctl", "mmap", "munmap",
                    "sendto", "recvfrom", "sendmsg", "recvmsg", "clone",
                    "epoll_pwait", "poll", "close", "gettid",
                    "newfstatat", "rt_sigprocmask"]

    rows = []
    for name in key_syscalls:
        n_data = n_lookup.get(name, {})
        c_data = c_lookup.get(name, {})
        n_count = n_data.get("count", 0)
        c_count = c_data.get("count", 0)
        n_lat = n_data.get("avg_latency_ns", 0)
        c_lat = c_data.get("avg_latency_ns", 0)

        if n_count > 0 or c_count > 0:
            count_overhead = ((c_count - n_count) / n_count * 100) if n_count > 0 else float("inf")
            lat_overhead = ((c_lat - n_lat) / n_lat * 100) if n_lat > 0 else float("inf")
            rows.append([name, f"{n_count:,}", f"{c_count:,}",
                         f"{count_overhead:+.1f}%",
                         f"{n_lat/1000:.1f}", f"{c_lat/1000:.1f}",
                         f"{lat_overhead:+.1f}%"])

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.4 + 2)))
    ax.axis("off")
    col_labels = ["Syscall", "Native\nCount", "Container\nCount", "Count\nOverhead",
                  "Native\nAvg (μs)", "Container\nAvg (μs)", "Latency\nOverhead"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#E3F2FD"] * 7)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Color overhead cells
    for i, row in enumerate(rows):
        for j in [3, 6]:  # overhead columns
            cell = table[i + 1, j]
            val = row[j]
            if "+" in val and "inf" not in val:
                cell.set_facecolor("#FFCDD2")  # red
            elif "-" in val:
                cell.set_facecolor("#C8E6C9")  # green

    ax.set_title("Per-Syscall Overhead: Native vs Containerized",
                 fontsize=14, fontweight="bold", pad=20)
    plt.savefig(os.path.join(output_dir, "compare_syscall_table.png"))
    plt.close()
    print("  Saved compare_syscall_table.png")


def plot_gpu_comparison(native_gpu, container_gpu, output_dir):
    """Plot 5: GPU utilization and power timeline comparison."""
    if not native_gpu or not container_gpu:
        print("  Skipping GPU comparison (missing data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, (data, label, color) in enumerate([
        (native_gpu, "Native", NATIVE_COLOR),
        (container_gpu, "Container", CONTAINER_COLOR),
    ]):
        # Filter GPU 0
        gpu0 = [r for r in data if r.get("gpu_index", 0) == 0]
        if not gpu0:
            gpu0 = data

        timestamps = [r.get("timestamp", 0) for r in gpu0]
        if timestamps:
            t0 = min(timestamps)
            times = [(t - t0) for t in timestamps]
        else:
            continue

        utils = [r.get("gpu_util_pct", 0) for r in gpu0]
        power = [r.get("power_w", 0) for r in gpu0]
        mem = [r.get("mem_used_mib", 0) for r in gpu0]
        temp = [r.get("temperature_c", 0) for r in gpu0]

        # GPU Utilization
        axes[0][idx].plot(times, utils, color=color, alpha=0.8, linewidth=0.8)
        axes[0][idx].fill_between(times, utils, alpha=0.2, color=color)
        axes[0][idx].set_ylabel("GPU Utilization (%)")
        axes[0][idx].set_title(f"{label} — GPU Utilization")
        axes[0][idx].set_ylim(0, 105)
        axes[0][idx].grid(alpha=0.3)
        axes[0][idx].set_xlabel("Time (s)")

        # Power draw
        axes[1][idx].plot(times, power, color="red", alpha=0.8, linewidth=0.8)
        axes[1][idx].fill_between(times, power, alpha=0.15, color="red")
        axes[1][idx].set_ylabel("Power Draw (W)")
        axes[1][idx].set_title(f"{label} — Power Consumption")
        axes[1][idx].grid(alpha=0.3)
        axes[1][idx].set_xlabel("Time (s)")

    plt.suptitle("GPU Metrics: Native vs Containerized (H100 NVL)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_gpu_timeline.png"))
    plt.close()
    print("  Saved compare_gpu_timeline.png")


def plot_gpu_summary_bars(native_gpu, container_gpu, output_dir):
    """Plot 6: GPU summary statistics bars."""
    if not native_gpu or not container_gpu:
        return

    def gpu_stats(data):
        gpu0 = [r for r in data if r.get("gpu_index", 0) == 0]
        if not gpu0:
            gpu0 = data
        utils = [r.get("gpu_util_pct", 0) for r in gpu0 if isinstance(r.get("gpu_util_pct", 0), (int, float))]
        power = [r.get("power_w", 0) for r in gpu0 if isinstance(r.get("power_w", 0), (int, float))]
        mem = [r.get("mem_used_mib", 0) for r in gpu0 if isinstance(r.get("mem_used_mib", 0), (int, float))]
        temp = [r.get("temperature_c", 0) for r in gpu0 if isinstance(r.get("temperature_c", 0), (int, float))]
        return {
            "avg_util": np.mean(utils) if utils else 0,
            "max_util": np.max(utils) if utils else 0,
            "avg_power": np.mean(power) if power else 0,
            "max_power": np.max(power) if power else 0,
            "peak_mem": np.max(mem) if mem else 0,
            "max_temp": np.max(temp) if temp else 0,
        }

    n_stats = gpu_stats(native_gpu)
    c_stats = gpu_stats(container_gpu)

    metrics = ["avg_util", "max_util", "avg_power", "max_power", "peak_mem", "max_temp"]
    labels = ["Avg GPU\nUtil (%)", "Peak GPU\nUtil (%)", "Avg Power\n(W)", "Peak Power\n(W)", "Peak Mem\n(MiB)", "Peak Temp\n(°C)"]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(metrics))
    width = 0.35

    n_vals = [n_stats[m] for m in metrics]
    c_vals = [c_stats[m] for m in metrics]

    bars1 = ax.bar(x - width/2, n_vals, width, label="Native", color=NATIVE_COLOR, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, c_vals, width, label="Container", color=CONTAINER_COLOR, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    f"{height:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("GPU Summary: Native vs Containerized", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_gpu_summary.png"))
    plt.close()
    print("  Saved compare_gpu_summary.png")


def plot_network_comparison(native_net, container_net, output_dir):
    """Plot 7: Network latency distribution comparison."""
    if not native_net or not container_net:
        print("  Skipping network comparison (missing data)")
        return

    def extract_latencies(data, event_type):
        return [r["latency_ns"] / 1000.0 for r in data
                if r.get("event_type") == event_type and r.get("latency_ns", 0) > 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) TCP Send latency CDF
    n_send = extract_latencies(native_net, "tcp_send")
    c_send = extract_latencies(container_net, "tcp_send")

    if n_send and c_send:
        for data, label, color in [(n_send, "Native", NATIVE_COLOR), (c_send, "Container", CONTAINER_COLOR)]:
            sorted_data = np.sort(data)
            # Clip to 99th percentile for visualization
            p99 = np.percentile(sorted_data, 99)
            clipped = sorted_data[sorted_data <= p99]
            cdf = np.arange(1, len(clipped) + 1) / len(sorted_data)
            axes[0].plot(clipped, cdf, label=label, color=color, linewidth=2)
        axes[0].set_xlabel("Latency (μs)")
        axes[0].set_ylabel("CDF")
        axes[0].set_title("(a) TCP Send Latency CDF")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # (b) TCP Recv latency CDF
    n_recv = extract_latencies(native_net, "tcp_recv")
    c_recv = extract_latencies(container_net, "tcp_recv")

    if n_recv and c_recv:
        for data, label, color in [(n_recv, "Native", NATIVE_COLOR), (c_recv, "Container", CONTAINER_COLOR)]:
            sorted_data = np.sort(data)
            p99 = np.percentile(sorted_data, 99)
            clipped = sorted_data[sorted_data <= p99]
            cdf = np.arange(1, len(clipped) + 1) / len(sorted_data)
            axes[1].plot(clipped, cdf, label=label, color=color, linewidth=2)
        axes[1].set_xlabel("Latency (μs)")
        axes[1].set_ylabel("CDF")
        axes[1].set_title("(b) TCP Recv Latency CDF")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    # (c) Box plot comparison
    box_data = []
    box_labels = []
    box_colors = []
    for data, label, color, etype in [
        (native_net, "Native\nSend", NATIVE_COLOR, "tcp_send"),
        (container_net, "Container\nSend", CONTAINER_COLOR, "tcp_send"),
        (native_net, "Native\nRecv", NATIVE_COLOR, "tcp_recv"),
        (container_net, "Container\nRecv", CONTAINER_COLOR, "tcp_recv"),
    ]:
        lats = extract_latencies(data, etype)
        if lats:
            # Clip outliers for box plot
            p99 = np.percentile(lats, 99)
            box_data.append([x for x in lats if x <= p99])
            box_labels.append(label)
            box_colors.append(color)

    if box_data:
        bp = axes[2].boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[2].set_ylabel("Latency (μs)")
        axes[2].set_title("(c) Latency Distribution")
        axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Network Stack Overhead: Native vs Containerized",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_network.png"))
    plt.close()
    print("  Saved compare_network.png")


def plot_network_stats_bars(native_net, container_net, output_dir):
    """Plot 8: Network summary statistics."""
    if not native_net or not container_net:
        return

    def net_stats(data):
        send_lats = [r["latency_ns"] for r in data if r.get("event_type") == "tcp_send" and r.get("latency_ns", 0) > 0]
        recv_lats = [r["latency_ns"] for r in data if r.get("event_type") == "tcp_recv" and r.get("latency_ns", 0) > 0]
        send_bytes = sum(r.get("bytes", 0) for r in data if r.get("event_type") == "tcp_send")
        recv_bytes = sum(r.get("bytes", 0) for r in data if r.get("event_type") == "tcp_recv")
        return {
            "send_count": len(send_lats),
            "recv_count": len(recv_lats),
            "send_avg_us": np.mean(send_lats) / 1000 if send_lats else 0,
            "recv_avg_us": np.mean(recv_lats) / 1000 if recv_lats else 0,
            "send_p99_us": np.percentile(send_lats, 99) / 1000 if send_lats else 0,
            "recv_p99_us": np.percentile(recv_lats, 99) / 1000 if recv_lats else 0,
            "send_bytes_mb": send_bytes / 1e6,
            "recv_bytes_mb": recv_bytes / 1e6,
        }

    n = net_stats(native_net)
    c = net_stats(container_net)

    metrics = ["send_count", "recv_count", "send_avg_us", "recv_avg_us", "send_p99_us", "recv_p99_us"]
    labels = ["Send\nCount", "Recv\nCount", "Send Avg\n(μs)", "Recv Avg\n(μs)", "Send P99\n(μs)", "Recv P99\n(μs)"]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(metrics))
    width = 0.35

    n_vals = [n[m] for m in metrics]
    c_vals = [c[m] for m in metrics]

    ax.bar(x - width/2, n_vals, width, label="Native", color=NATIVE_COLOR, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, c_vals, width, label="Container", color=CONTAINER_COLOR, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Network Statistics: Native vs Containerized", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_network_stats.png"))
    plt.close()
    print("  Saved compare_network_stats.png")


def plot_scheduling_latency_comparison(native_lats, container_lats, output_dir):
    """Plot 9: Scheduling latency CDF and histogram."""
    if not native_lats or not container_lats:
        print("  Skipping scheduling latency comparison (missing data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) CDF
    for data, label, color in [(native_lats, "Native", NATIVE_COLOR),
                                (container_lats, "Container", CONTAINER_COLOR)]:
        arr = np.array(data, dtype=np.float64) / 1000.0  # to μs
        sorted_arr = np.sort(arr)
        p99 = np.percentile(sorted_arr, 99)
        clipped = sorted_arr[sorted_arr <= p99]
        cdf = np.arange(1, len(clipped) + 1) / len(sorted_arr)
        axes[0].plot(clipped, cdf, label=f"{label} (n={len(arr):,})", color=color, linewidth=2)

    axes[0].set_xlabel("Scheduling Latency (μs)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("(a) Scheduling Latency CDF (99th percentile)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # (b) Histogram
    n_arr = np.array(native_lats, dtype=np.float64) / 1000.0
    c_arr = np.array(container_lats, dtype=np.float64) / 1000.0
    p99_max = max(np.percentile(n_arr, 99), np.percentile(c_arr, 99))
    bins = np.linspace(0, p99_max, 100)

    axes[1].hist(n_arr[n_arr <= p99_max], bins=bins, alpha=0.5, label="Native",
                 color=NATIVE_COLOR, density=True, edgecolor="black", linewidth=0.3)
    axes[1].hist(c_arr[c_arr <= p99_max], bins=bins, alpha=0.5, label="Container",
                 color=CONTAINER_COLOR, density=True, edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Scheduling Latency (μs)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("(b) Scheduling Latency Distribution")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("CPU Scheduling Latency: Native vs Containerized",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_sched_latency.png"))
    plt.close()
    print("  Saved compare_sched_latency.png")


def plot_overhead_summary(native_json, container_json, native_syscalls, container_syscalls,
                          native_net, container_net, native_lats, container_lats,
                          native_gpu, container_gpu, output_dir):
    """Plot 10: Final overhead attribution summary — THE KEY DELIVERABLE."""

    overhead_data = {}

    # Training time
    if native_json and container_json:
        n_time = native_json.get("total_time_sec", 0)
        c_time = container_json.get("total_time_sec", 0)
        if n_time > 0:
            overhead_data["Training Time"] = ((c_time - n_time) / n_time) * 100

    # Syscall overhead
    if native_syscalls and container_syscalls:
        n_total = sum(r.get("count", 0) for r in native_syscalls)
        c_total = sum(r.get("count", 0) for r in container_syscalls)
        if n_total > 0:
            overhead_data["Syscall Count"] = ((c_total - n_total) / n_total) * 100

        # Weighted average latency
        n_weighted = sum(r.get("count", 0) * r.get("avg_latency_ns", 0) for r in native_syscalls)
        c_weighted = sum(r.get("count", 0) * r.get("avg_latency_ns", 0) for r in container_syscalls)
        if n_weighted > 0:
            overhead_data["Syscall Latency\n(weighted)"] = ((c_weighted - n_weighted) / n_weighted) * 100

    # Network overhead
    if native_net and container_net:
        n_send = [r["latency_ns"] for r in native_net if r.get("event_type") == "tcp_send" and r.get("latency_ns", 0) > 0]
        c_send = [r["latency_ns"] for r in container_net if r.get("event_type") == "tcp_send" and r.get("latency_ns", 0) > 0]
        if n_send and c_send:
            overhead_data["TCP Send\nLatency"] = ((np.mean(c_send) - np.mean(n_send)) / np.mean(n_send)) * 100
        n_recv = [r["latency_ns"] for r in native_net if r.get("event_type") == "tcp_recv" and r.get("latency_ns", 0) > 0]
        c_recv = [r["latency_ns"] for r in container_net if r.get("event_type") == "tcp_recv" and r.get("latency_ns", 0) > 0]
        if n_recv and c_recv:
            overhead_data["TCP Recv\nLatency"] = ((np.mean(c_recv) - np.mean(n_recv)) / np.mean(n_recv)) * 100

    # Scheduling latency
    if native_lats and container_lats:
        n_mean = np.mean(native_lats)
        c_mean = np.mean(container_lats)
        if n_mean > 0:
            overhead_data["Sched Latency\n(mean)"] = ((c_mean - n_mean) / n_mean) * 100

    if not overhead_data:
        print("  Skipping overhead summary (insufficient data)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(overhead_data.keys())
    values = list(overhead_data.values())
    colors = ["#EF5350" if v > 0 else "#66BB6A" for v in values]

    bars = ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.5, height=0.6)

    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        ha = "left" if val >= 0 else "right"
        offset = 1 if val >= 0 else -1
        ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", ha=ha, va="center", fontweight="bold", fontsize=11)

    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlabel("Container Overhead (%)")
    ax.set_title("Per-Subsystem Containerization Overhead\n(Positive = Container is Slower)",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overhead_attribution.png"))
    plt.close()
    print("  Saved overhead_attribution.png")


def generate_summary_table(native_json, container_json, native_syscalls, container_syscalls,
                           native_net, container_net, native_lats, container_lats,
                           native_gpu, container_gpu, output_dir):
    """Generate the final per-subsystem overhead table (Objective 1 deliverable)."""

    rows = []

    # Training metrics
    if native_json and container_json:
        n_time = native_json.get("total_time_sec", 0)
        c_time = container_json.get("total_time_sec", 0)
        overhead = ((c_time - n_time) / n_time * 100) if n_time > 0 else 0
        rows.append(["Training Time (sec)", f"{n_time:.1f}", f"{c_time:.1f}", f"{overhead:+.1f}%"])

        n_epochs = native_json.get("epochs", [])
        c_epochs = container_json.get("epochs", [])
        if n_epochs and c_epochs:
            n_tput = np.mean([e["samples_per_sec"] for e in n_epochs])
            c_tput = np.mean([e["samples_per_sec"] for e in c_epochs])
            overhead = ((c_tput - n_tput) / n_tput * 100) if n_tput > 0 else 0
            rows.append(["Throughput (samples/sec)", f"{n_tput:.0f}", f"{c_tput:.0f}", f"{overhead:+.1f}%"])

    # Scheduling latency
    if native_lats and container_lats:
        n_arr = np.array(native_lats, dtype=np.float64)
        c_arr = np.array(container_lats, dtype=np.float64)
        for label, func in [("Sched Latency Mean (μs)", np.mean),
                            ("Sched Latency P95 (μs)", lambda x: np.percentile(x, 95)),
                            ("Sched Latency P99 (μs)", lambda x: np.percentile(x, 99))]:
            nv = func(n_arr) / 1000
            cv = func(c_arr) / 1000
            overhead = ((cv - nv) / nv * 100) if nv > 0 else 0
            rows.append([label, f"{nv:.2f}", f"{cv:.2f}", f"{overhead:+.1f}%"])

    # Syscall metrics
    if native_syscalls and container_syscalls:
        n_total = sum(r.get("count", 0) for r in native_syscalls)
        c_total = sum(r.get("count", 0) for r in container_syscalls)
        overhead = ((c_total - n_total) / n_total * 100) if n_total > 0 else 0
        rows.append(["Total Syscalls", f"{n_total:,}", f"{c_total:,}", f"{overhead:+.1f}%"])
        rows.append(["Unique Syscall Types", f"{len(native_syscalls)}", f"{len(container_syscalls)}", ""])

    # Network metrics
    if native_net and container_net:
        for etype, label in [("tcp_send", "TCP Send"), ("tcp_recv", "TCP Recv")]:
            n_lats = [r["latency_ns"] for r in native_net if r.get("event_type") == etype and r.get("latency_ns", 0) > 0]
            c_lats = [r["latency_ns"] for r in container_net if r.get("event_type") == etype and r.get("latency_ns", 0) > 0]
            if n_lats and c_lats:
                n_avg = np.mean(n_lats) / 1000
                c_avg = np.mean(c_lats) / 1000
                overhead = ((c_avg - n_avg) / n_avg * 100) if n_avg > 0 else 0
                rows.append([f"{label} Avg Latency (μs)", f"{n_avg:.2f}", f"{c_avg:.2f}", f"{overhead:+.1f}%"])
                rows.append([f"{label} Count", f"{len(n_lats):,}", f"{len(c_lats):,}", ""])

    # GPU metrics
    if native_gpu and container_gpu:
        for data, label_prefix in [(native_gpu, "n_"), (container_gpu, "c_")]:
            gpu0 = [r for r in data if r.get("gpu_index", 0) == 0]
            utils = [r.get("gpu_util_pct", 0) for r in gpu0 if isinstance(r.get("gpu_util_pct", 0), (int, float))]
            power = [r.get("power_w", 0) for r in gpu0 if isinstance(r.get("power_w", 0), (int, float))]
            if label_prefix == "n_":
                n_util = np.mean(utils) if utils else 0
                n_power = np.mean(power) if power else 0
            else:
                c_util = np.mean(utils) if utils else 0
                c_power = np.mean(power) if power else 0

        if n_util > 0:
            overhead = ((c_util - n_util) / n_util * 100)
            rows.append(["GPU Avg Utilization (%)", f"{n_util:.1f}", f"{c_util:.1f}", f"{overhead:+.1f}%"])
        if n_power > 0:
            overhead = ((c_power - n_power) / n_power * 100)
            rows.append(["GPU Avg Power (W)", f"{n_power:.1f}", f"{c_power:.1f}", f"{overhead:+.1f}%"])

    if not rows:
        return

    # Create table figure
    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.4 + 2)))
    ax.axis("off")
    col_labels = ["Metric", "Native", "Container", "Overhead"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#1565C0", "#1565C0", "#1565C0", "#1565C0"])

    # Style header
    for j in range(4):
        table[0, j].set_text_props(color="white", fontweight="bold")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color overhead cells
    for i, row in enumerate(rows):
        overhead_str = row[3]
        cell = table[i + 1, 3]
        if "+" in overhead_str:
            cell.set_facecolor("#FFCDD2")
        elif "-" in overhead_str:
            cell.set_facecolor("#C8E6C9")
        else:
            cell.set_facecolor("#F5F5F5")

    ax.set_title("Per-Subsystem Overhead: Native vs Containerized\n(Objective 1 Deliverable)",
                 fontsize=15, fontweight="bold", pad=20)
    plt.savefig(os.path.join(output_dir, "overhead_summary_table.png"))
    plt.close()
    print("  Saved overhead_summary_table.png")

    # Also save as JSON
    json_rows = []
    for row in rows:
        json_rows.append({"metric": row[0], "native": row[1], "container": row[2], "overhead": row[3]})
    with open(os.path.join(output_dir, "overhead_summary.json"), "w") as f:
        json.dump(json_rows, f, indent=2)
    print("  Saved overhead_summary.json")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Results Comparison - Group 21"
    )
    parser.add_argument("--native", type=str, default="results/native",
                        help="Native results directory")
    parser.add_argument("--container", type=str, default="results/container",
                        help="Container results directory")
    parser.add_argument("--output-dir", type=str, default="results/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("Native vs Containerized ML Workload Profiling")
    print("Group 21 — GRS Project Part A")
    print("=" * 70)

    # Load all data
    print("\nLoading data...")

    native_json = load_training_json(os.path.join(args.native, "21_training_native.json"))
    container_json = load_training_json(os.path.join(args.container, "21_training_container.json"))
    print(f"  Training JSON: native={'OK' if native_json else 'MISSING'}, container={'OK' if container_json else 'MISSING'}")

    native_syscalls = load_syscall_csv(os.path.join(args.native, "21_syscall_results.csv"))
    container_syscalls = load_syscall_csv(os.path.join(args.container, "21_syscall_results.csv"))
    print(f"  Syscalls: native={len(native_syscalls)} types, container={len(container_syscalls)} types")

    native_gpu = load_gpu_csv(os.path.join(args.native, "21_gpu_results.csv"))
    container_gpu = load_gpu_csv(os.path.join(args.container, "21_gpu_results.csv"))
    print(f"  GPU samples: native={len(native_gpu)}, container={len(container_gpu)}")

    native_net = load_net_csv(os.path.join(args.native, "21_net_results.csv"))
    container_net = load_net_csv(os.path.join(args.container, "21_net_results.csv"))
    print(f"  Network events: native={len(native_net)}, container={len(container_net)}")

    print("  Loading scheduling latency data (may take a moment for large files)...")
    native_lats = load_sched_latency_csv(os.path.join(args.native, "21_cpu_results_sched_latency.csv"))
    container_lats = load_sched_latency_csv(os.path.join(args.container, "21_cpu_results_sched_latency.csv"))
    print(f"  Sched latency: native={len(native_lats):,} events, container={len(container_lats):,} events")

    # Generate all plots
    print(f"\nGenerating comparison plots in {output_dir}/...")

    plot_training_comparison(native_json, container_json, output_dir)
    plot_total_time_comparison(native_json, container_json, output_dir)
    plot_syscall_comparison(native_syscalls, container_syscalls, output_dir)
    plot_syscall_overhead_table(native_syscalls, container_syscalls, output_dir)
    plot_gpu_comparison(native_gpu, container_gpu, output_dir)
    plot_gpu_summary_bars(native_gpu, container_gpu, output_dir)
    plot_network_comparison(native_net, container_net, output_dir)
    plot_network_stats_bars(native_net, container_net, output_dir)
    plot_scheduling_latency_comparison(native_lats, container_lats, output_dir)
    plot_overhead_summary(native_json, container_json, native_syscalls, container_syscalls,
                          native_net, container_net, native_lats, container_lats,
                          native_gpu, container_gpu, output_dir)
    generate_summary_table(native_json, container_json, native_syscalls, container_syscalls,
                           native_net, container_net, native_lats, container_lats,
                           native_gpu, container_gpu, output_dir)

    print(f"\n{'=' * 70}")
    print(f"ALL PLOTS GENERATED — {output_dir}/")
    print(f"{'=' * 70}")
    print(f"\nPlot files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".png") or f.endswith(".json"):
            fpath = os.path.join(output_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
