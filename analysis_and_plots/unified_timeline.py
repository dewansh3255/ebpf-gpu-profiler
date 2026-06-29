import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import os

def load_network_data(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['Payload_MB_or_Delay_MS'] = pd.to_numeric(df['Payload_MB_or_Delay_MS'], errors='coerce')
    return df.dropna(subset=['Payload_MB_or_Delay_MS'])

def load_gpu_data(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['Duration_ms'] = pd.to_numeric(df['Duration_ms'], errors='coerce')
    return df.dropna(subset=['Duration_ms'])

def generate_unified_timeline(net_csv, gpu_csv):
    print("Loading Phase 1 (Network/CPU) and Phase 2 (GPU) data...")
    df_net = load_network_data(net_csv)
    df_gpu = load_gpu_data(gpu_csv)

    if df_net.empty or df_gpu.empty:
        print("[!] Error: One or both CSV files are empty or missing.")
        sys.exit(1)

    # Set up the visualization
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Unified System Profiling: CPU, Network, and GPU Timelines", fontsize=18, fontweight='bold')

    # --- ROW 1: GPU Hardware & PCIe Bus ---
    ax_gpu = axes[0]
    sns.scatterplot(data=df_gpu, x='Time', y='Duration_ms', hue='Event_Type', 
                    palette={'COMPUTE_MATH': '#2ecc71', 'MEM_TRANSFER': '#f39c12'}, 
                    ax=ax_gpu, s=50, alpha=0.7, edgecolor=None)
    ax_gpu.set_title("Layer 1: GPU Compute & PCIe Memory Transfers", fontweight='bold')
    ax_gpu.set_ylabel("Duration (ms)")
    
    # Highlight the "GPU Idle" gaps
    ax_gpu.annotate('<- GPU IDLE WAITING FOR NETWORK ->', 
                    xy=(0.5, 0.8), xycoords='axes fraction', 
                    ha='center', fontsize=12, color='red', alpha=0.5)

    # --- ROW 2: CPU Scheduler ---
    ax_cpu = axes[1]
    df_sched = df_net[df_net['Event_Type'] == 'SCHED_DELAY']
    if not df_sched.empty:
        sns.lineplot(data=df_sched, x='Time', y='Payload_MB_or_Delay_MS', 
                     color='#9b59b6', ax=ax_cpu, linewidth=2, marker="o")
    ax_cpu.set_title("Layer 2: CPU Scheduler Delay (Wakeup Latency)", fontweight='bold')
    ax_cpu.set_ylabel("Delay (ms)")

    # --- ROW 3: Network I/O and Drops ---
    ax_net = axes[2]
    df_io = df_net[df_net['Event_Type'] == 'NETWORK_IO']
    df_drop = df_net[df_net['Event_Type'] == 'TCP_RETRANSMIT']
    
    if not df_io.empty:
        sns.scatterplot(data=df_io, x='Time', y='Payload_MB_or_Delay_MS', hue='Direction',
                        palette={'IN_RECV': '#3498db', 'OUT_SEND': '#e74c3c'}, 
                        ax=ax_net, s=40, alpha=0.6, edgecolor=None)
    
    if not df_drop.empty:
        # Plot drops as large red 'X' marks on the bottom of the graph
        ax_net.scatter(df_drop['Time'], [0.1] * len(df_drop), 
                       color='red', marker='x', s=100, label='TCP DROP (Retransmit)', linewidths=2)

    ax_net.set_title("Layer 3: Wi-Fi Network I/O & Packet Drops", fontweight='bold')
    ax_net.set_ylabel("Payload (MB)")
    ax_net.set_xlabel("Timeline (HH:MM:SS)", fontsize=12, fontweight='bold')

    # Format the X-axis to show clean time strings
    ax_net.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    # Prevent overlapping labels
    plt.tight_layout()
    
    output_filename = "phase3_unified_timeline.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[✓] Dashboard generated successfully: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_unified_timeline.py <PHASE1_NETWORK_CSV> <PHASE2_GPU_CSV>")
        sys.exit(1)
    
    generate_unified_timeline(sys.argv[1], sys.argv[2])