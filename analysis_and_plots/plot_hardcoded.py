#!/usr/bin/env python3
"""
G_21_plot_hardcoded.py - Hardcoded matplotlib plots for eBPF/eGPU experiment results.
Group 21 - Part B Final Submission

All data values are hardcoded from actual experiment results.
No CSV files are read as input.

Experiment: ResNet-18 on CIFAR-10, 2x NVIDIA H100 NVL GPUs, 10 epochs, batch_size=128
Comparison: Native vs Docker Container execution with eBPF profiling
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots_hardcoded")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_training_loss():
    """Plot 1: Training Loss per Epoch - Native vs Container"""
    epochs = list(range(1, 11))
    native_loss = [1.6631, 1.2166, 0.9949, 0.8560, 0.7412, 0.6691, 0.6013, 0.5540, 0.5202, 0.5151]
    container_loss = [1.6816, 1.2233, 1.0060, 0.8669, 0.7425, 0.6634, 0.6050, 0.5505, 0.5198, 0.5118]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, native_loss, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Native')
    ax.plot(epochs, container_loss, 's--', color='#FF5722', linewidth=2, markersize=8, label='Container')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Training Loss', fontsize=13)
    ax.set_title('Training Loss: Native vs Container', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_training_loss.png"), dpi=150)
    plt.close(fig)
    print("  [1/11] Training Loss plot saved")


def plot_training_accuracy():
    """Plot 2: Training & Test Accuracy per Epoch"""
    epochs = list(range(1, 11))
    native_train_acc = [37.95, 55.72, 64.21, 69.63, 73.93, 76.19, 79.07, 80.25, 81.72, 81.84]
    container_train_acc = [37.31, 55.60, 63.94, 69.06, 73.80, 76.54, 78.87, 80.72, 81.63, 81.89]
    native_test_acc = [47.03, 57.65, 67.59, 67.06, 70.36, 72.84, 76.99, 79.45, 80.49, 81.14]
    container_test_acc = [46.78, 56.40, 61.89, 61.89, 71.20, 74.53, 77.46, 79.31, 80.78, 81.06]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, native_train_acc, 'o-', color='#2196F3', linewidth=2, markersize=7, label='Native')
    ax1.plot(epochs, container_train_acc, 's--', color='#FF5722', linewidth=2, markersize=7, label='Container')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax1.set_title('Training Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    ax2.plot(epochs, native_test_acc, 'o-', color='#2196F3', linewidth=2, markersize=7, label='Native')
    ax2.plot(epochs, container_test_acc, 's--', color='#FF5722', linewidth=2, markersize=7, label='Container')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    fig.suptitle('Model Accuracy: Native vs Container', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_training_accuracy.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [2/11] Training Accuracy plot saved")


def plot_throughput():
    """Plot 3: Throughput (samples/sec) per Epoch"""
    epochs = list(range(1, 11))
    native_throughput = [8023, 9919, 9863, 9887, 9794, 8968, 9201, 7826, 9931, 9323]
    container_throughput = [7740, 8491, 8953, 8944, 8519, 8396, 8749, 8538, 7710, 8676]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(epochs))
    width = 0.35
    bars1 = ax.bar(x - width/2, native_throughput, width, label='Native', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, container_throughput, width, label='Container', color='#FF5722', alpha=0.85)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Throughput (samples/sec)', fontsize=13)
    ax.set_title('Training Throughput: Native vs Container', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add averages
    native_avg = np.mean(native_throughput)
    container_avg = np.mean(container_throughput)
    ax.axhline(y=native_avg, color='#2196F3', linestyle=':', alpha=0.7, label=f'Native avg: {native_avg:.0f}')
    ax.axhline(y=container_avg, color='#FF5722', linestyle=':', alpha=0.7, label=f'Container avg: {container_avg:.0f}')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_throughput.png"), dpi=150)
    plt.close(fig)
    print("  [3/11] Throughput plot saved")


def plot_epoch_time():
    """Plot 4: Per-Epoch Training Time"""
    epochs = list(range(1, 11))
    native_time = [3.12, 2.52, 2.53, 2.53, 2.55, 2.79, 2.72, 3.19, 2.52, 2.68]
    container_time = [3.23, 2.94, 2.79, 2.79, 2.93, 2.98, 2.86, 2.93, 3.24, 2.88]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, native_time, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Native')
    ax.plot(epochs, container_time, 's--', color='#FF5722', linewidth=2, markersize=8, label='Container')

    ax.axhline(y=np.mean(native_time), color='#2196F3', linestyle=':', alpha=0.6,
               label=f'Native avg: {np.mean(native_time):.2f}s')
    ax.axhline(y=np.mean(container_time), color='#FF5722', linestyle=':', alpha=0.6,
               label=f'Container avg: {np.mean(container_time):.2f}s')

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Time (seconds)', fontsize=13)
    ax.set_title('Per-Epoch Training Time: Native vs Container', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_epoch_time.png"), dpi=150)
    plt.close(fig)
    print("  [4/11] Epoch Time plot saved")


def plot_total_time_comparison():
    """Plot 5: Total Training Time + Overhead Summary"""
    categories = ['Training\nTime (s)', 'Throughput\n(avg samp/s)', 'GPU 0 Util\n(active %)', 'GPU 0 Power\n(avg W)']
    native_vals = [31.8, 9274, 76.7, 158.4]
    container_vals = [34.3, 8472, 74.1, 198.9]
    overhead_pct = [7.9, -8.6, -3.4, 25.6]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for i, (cat, nv, cv, oh) in enumerate(zip(categories, native_vals, container_vals, overhead_pct)):
        ax = axes[i]
        bars = ax.bar(['Native', 'Container'], [nv, cv],
                      color=['#2196F3', '#FF5722'], alpha=0.85, width=0.5)
        ax.set_title(cat, fontsize=11, fontweight='bold')
        for bar, val in zip(bars, [nv, cv]):
            fmt = f'{val:.1f}' if isinstance(val, float) else f'{val:,}'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    fmt, ha='center', va='bottom', fontsize=10, fontweight='bold')
        oh_color = '#d32f2f' if oh > 0 else '#388e3c'
        oh_sign = '+' if oh >= 0 else ''
        ax.text(0.5, 0.02, f'Overhead: {oh_sign}{oh}%', transform=ax.transAxes,
                ha='center', fontsize=10, color=oh_color, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Overall Performance: Native vs Container', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_total_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [5/11] Total Comparison plot saved")


def plot_gpu_utilization():
    """Plot 6: GPU Utilization, Power & Temperature comparison (bar summary)"""
    gpu_labels = ['GPU 0\n(Native)', 'GPU 1\n(Native)', 'GPU 0\n(Container)', 'GPU 1\n(Container)']
    util_avg = [36.8, 39.0, 53.3, 56.5]
    util_max = [97, 95, 96, 93]
    power_avg = [158.4, 185.0, 198.9, 223.8]
    power_max = [285, 326, 276, 312]
    temp_avg = [39.0, 56.4, 45.4, 66.0]
    temp_max = [49, 80, 53, 82]

    colors = ['#2196F3', '#64B5F6', '#FF5722', '#FF8A65']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # GPU Utilization
    ax = axes[0]
    x = np.arange(4)
    bars = ax.bar(x, util_avg, color=colors, alpha=0.85, width=0.6)
    for i, (bar, mx) in enumerate(zip(bars, util_max)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'avg:{util_avg[i]:.1f}%\nmax:{mx}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Power
    ax = axes[1]
    bars = ax.bar(x, power_avg, color=colors, alpha=0.85, width=0.6)
    for i, (bar, mx) in enumerate(zip(bars, power_max)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'avg:{power_avg[i]:.0f}W\nmax:{mx:.0f}W', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('GPU Power Draw', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Temperature
    ax = axes[2]
    bars = ax.bar(x, temp_avg, color=colors, alpha=0.85, width=0.6)
    for i, (bar, mx) in enumerate(zip(bars, temp_max)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'avg:{temp_avg[i]:.0f}°C\nmax:{mx}°C', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('GPU Temperature', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('GPU Metrics: Native vs Container (2× NVIDIA H100 NVL)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_gpu_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [6/11] GPU Metrics plot saved")


def plot_syscall_comparison():
    """Plot 7: Top Syscalls - Native vs Container"""
    syscall_names = ['gettid', 'poll', 'futex', 'read', 'recvmsg', 'write', 'close', 'newfstatat']
    native_counts = [5762441, 2973232, 712280, 548232, 251039, 204350, 198720, 187227]
    container_counts = [4011335, 1873974, 632465, 289317, 23010, 128231, 195910, 181479]

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(syscall_names))
    height = 0.35
    bars1 = ax.barh(y - height/2, [c/1e6 for c in native_counts], height,
                    label='Native', color='#2196F3', alpha=0.85)
    bars2 = ax.barh(y + height/2, [c/1e6 for c in container_counts], height,
                    label='Container', color='#FF5722', alpha=0.85)
    ax.set_xlabel('Count (millions)', fontsize=13)
    ax.set_ylabel('Syscall', fontsize=13)
    ax.set_title('Top Syscalls: Native vs Container', fontsize=15, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(syscall_names, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    native_total = 11926534
    container_total = 8180511
    native_rate = native_total / 61.1
    container_rate = container_total / 44.1
    ax.text(0.98, 0.02, f'Rate: Native={native_rate:,.0f}/s  Container={container_rate:,.0f}/s  |  Unique types: Native=121, Container=167',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_syscall_comparison.png"), dpi=150)
    plt.close(fig)
    print("  [7/11] Syscall Comparison plot saved")


def plot_syscall_latency():
    """Plot 8: Syscall Average Latency Comparison (non-blocking only)"""
    plot_names = ['gettid', 'read', 'close', 'newfstatat', 'write', 'openat', 'ioctl']
    native_lat = [0.7, 71.1, 1.3, 3.2, 9.5, 117.4, 117.0]
    container_lat = [0.8, 161.6, 1.3, 4.7, 5.9, 84.5, 107.7]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(plot_names))
    height = 0.35
    ax.barh(y - height/2, native_lat, height, label='Native', color='#2196F3', alpha=0.85)
    ax.barh(y + height/2, container_lat, height, label='Container', color='#FF5722', alpha=0.85)
    ax.set_xlabel('Average Latency (μs)', fontsize=13)
    ax.set_ylabel('Syscall', fontsize=13)
    ax.set_title('Syscall Average Latency (Non-Blocking): Native vs Container', fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(plot_names, fontsize=11)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_syscall_latency.png"), dpi=150)
    plt.close(fig)
    print("  [8/11] Syscall Latency plot saved")


def plot_network_comparison():
    """Plot 9: Network I/O - TCP Send/Recv Stats"""
    categories = ['TCP Send\nAvg Lat (μs)', 'TCP Recv\nAvg Lat (μs)', 'TCP Send\nCount', 'TCP Recv\nCount']
    native_vals = [30.4, 283.4, 6443, 110923]
    container_vals = [35.7, 6.8, 1014, 2732]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for i, (cat, nv, cv) in enumerate(zip(categories, native_vals, container_vals)):
        ax = axes[i]
        bars = ax.bar(['Native', 'Container'], [nv, cv],
                      color=['#2196F3', '#FF5722'], alpha=0.85, width=0.5)
        ax.set_title(cat, fontsize=11, fontweight='bold')
        for bar, val in zip(bars, [nv, cv]):
            fmt = f'{val:.1f}' if val < 100 else f'{val:,.0f}'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    fmt, ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Network I/O: Native vs Container (eBPF TCP Profiling)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "09_network_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [9/11] Network Comparison plot saved")


def plot_sched_latency():
    """Plot 10: CPU Scheduler Latency + Context Switches"""
    metrics = ['Mean\nLatency (μs)', 'P50\nLatency (μs)', 'P95\nLatency (μs)', 'P99\nLatency (μs)']
    native_vals = [13.9, 4.0, 15.2, 22.7]
    container_vals = [17.7, 3.6, 14.4, 21.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax1.bar(x - width/2, native_vals, width, label='Native', color='#2196F3', alpha=0.85)
    bars2 = ax1.bar(x + width/2, container_vals, width, label='Container', color='#FF5722', alpha=0.85)
    ax1.set_ylabel('Latency (μs)', fontsize=13)
    ax1.set_title('Scheduler Latency Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, native_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, container_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=9)

    # Context switches
    native_ctx = 3367166
    container_ctx = 2508117
    native_duration = 61.1
    container_duration = 43.6
    native_per_sec = native_ctx / native_duration
    container_per_sec = container_ctx / container_duration

    labels = ['Native', 'Container']
    totals = [native_ctx / 1e6, container_ctx / 1e6]
    per_sec = [native_per_sec, container_per_sec]

    bars = ax2.bar(labels, totals, color=['#2196F3', '#FF5722'], alpha=0.85, width=0.5)
    ax2.set_ylabel('Total Context Switches (millions)', fontsize=12)
    ax2.set_title('Context Switches (eBPF Tracepoints)', fontsize=13, fontweight='bold')
    for bar, total, ps in zip(bars, totals, per_sec):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{total:.2f}M\n({ps:,.0f}/s)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    native_rate_ctx = native_ctx / native_duration
    container_rate_ctx = container_ctx / container_duration
    rate_diff = (container_rate_ctx - native_rate_ctx) / native_rate_ctx * 100
    ax2.text(0.5, 0.02, f'Rate: Native={native_rate_ctx:,.0f}/s  Container={container_rate_ctx:,.0f}/s  ({rate_diff:+.1f}%)',
             transform=ax2.transAxes, ha='center', fontsize=10, color='#555555', fontweight='bold')

    fig.suptitle('CPU Scheduling: Native vs Container (eBPF Profiling)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "10_sched_latency.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [10/11] Scheduler Latency plot saved")


def plot_overhead_summary():
    """Plot 11: Overall Overhead Summary Table"""
    metrics = [
        'Training Time (sec)', 'Avg Throughput (samples/sec)',
        'Final Test Accuracy (%)', 'Sched Latency Mean (μs)',
        'Sched Latency P95 (μs)', 'Sched Latency P99 (μs)',
        'Context Switches (total)', 'Total Syscalls',
        'TCP Send Avg Lat (μs)', 'TCP Send Count',
        'TCP Recv Avg Lat (μs)', 'TCP Recv Count',
        'GPU 0 Avg Util (%)', 'GPU 0 Avg Power (W)',
        'GPU 0 Avg Temp (°C)'
    ]
    native_vals = ['31.8', '9,274', '81.1', '13.9', '15.2', '22.7',
                   '3,367,166', '11,926,534',
                   '30.4', '6,443', '283.4', '110,923',
                   '76.7', '158.4', '39.0']
    container_vals = ['34.3', '8,472', '81.1', '17.7', '14.4', '21.5',
                      '2,508,117', '8,180,511',
                      '35.7', '1,014', '6.8', '2,732',
                      '74.1', '198.9', '45.4']
    overheads = ['+7.9%', '-8.6%', '+0.0%', '+27.3%', '-5.3%', '-5.3%',
                 '-25.4%*', '-31.4%*',
                 '+17.4%', '-84.3%*', '-97.6%*', '-97.5%*',
                 '-3.4%', '+25.6%', '+16.4%']

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis('off')

    colors_overhead = []
    for oh in overheads:
        val_str = oh.replace('%', '').replace('+', '').replace('*', '')
        val = float(val_str)
        if abs(val) < 3:
            colors_overhead.append('#e8f5e9')
        elif val > 10:
            colors_overhead.append('#ffcdd2')
        elif val > 0:
            colors_overhead.append('#fff9c4')
        elif val < -10:
            colors_overhead.append('#c8e6c9')
        else:
            colors_overhead.append('#e8f5e9')

    cell_text = list(zip(metrics, native_vals, container_vals, overheads))
    table_data = [[m, n, c, o] for m, n, c, o in cell_text]
    col_labels = ['Metric', 'Native', 'Container', 'Overhead']

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center', colWidths=[0.35, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=11)

    # Color overhead cells
    for i, color in enumerate(colors_overhead):
        table[i + 1, 3].set_facecolor(color)

    # Alternate row colors
    for i in range(len(metrics)):
        base_color = '#f5f5f5' if i % 2 == 0 else 'white'
        for j in range(3):
            table[i + 1, j].set_facecolor(base_color)

    ax.set_title('Overhead Summary: Native vs Container\n'
                 'ResNet-18 / CIFAR-10 / 2×H100 NVL / 10 Epochs / eBPF Profiling\n'
                 '* = different profiling windows (native 61.1s vs container 43.6s); rates comparable',
                 fontsize=13, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "11_overhead_summary_table.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [11/11] Overhead Summary Table saved")


def main():
    print(f"Generating hardcoded plots to: {OUTPUT_DIR}")
    print("=" * 60)
    plot_training_loss()
    plot_training_accuracy()
    plot_throughput()
    plot_epoch_time()
    plot_total_time_comparison()
    plot_gpu_utilization()
    plot_syscall_comparison()
    plot_syscall_latency()
    plot_network_comparison()
    plot_sched_latency()
    plot_overhead_summary()
    print("=" * 60)
    print(f"All 11 plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
