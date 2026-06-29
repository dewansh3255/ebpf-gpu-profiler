import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_gpu_metrics(results_dir):
    file_path = os.path.join(results_dir, "21_gpu_results.csv")
    if not os.path.exists(file_path): return
    
    df = pd.read_csv(file_path)
    # Filter for GPU 0 (the active one)
    df = df[df['gpu_index'] == 0]
    
    # Normalize timestamp to start at 0
    df['time_sec'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Utilization (%)', color='tab:blue')
    ax1.plot(df['time_sec'], df['gpu_util_pct'], label='GPU Utilization', color='tab:blue')
    ax1.plot(df['time_sec'], df['mem_util_pct'], label='Memory Utilization', color='tab:cyan', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (W)', color='tab:red')
    ax2.plot(df['time_sec'], df['power_w'], label='Power Draw', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title(f'GPU 0 Utilization and Power Over Time ({results_dir})')
    fig.tight_layout()
    plt.savefig(f'{results_dir}_gpu_timeline.png')
    print(f"Saved {results_dir}_gpu_timeline.png")

def plot_syscall_counts(results_dir):
    file_path = os.path.join(results_dir, "21_syscall_results.csv")
    if not os.path.exists(file_path): return
    
    df = pd.read_csv(file_path)
    # Get top 15 syscalls by count
    top_syscalls = df.nlargest(15, 'count').sort_values('count', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_syscalls['name'], top_syscalls['count'], color='purple')
    plt.xlabel('Total Count')
    plt.title(f'Top 15 System Calls by Frequency ({results_dir})')
    plt.tight_layout()
    plt.savefig(f'{results_dir}_syscall_counts.png')
    print(f"Saved {results_dir}_syscall_counts.png")

def plot_network_latency(results_dir):
    file_path = os.path.join(results_dir, "21_net_results.csv")
    if not os.path.exists(file_path): return
    
    df = pd.read_csv(file_path)
    # Convert ns to us
    df['latency_us'] = df['latency_ns'] / 1000.0 
    
    plt.figure(figsize=(8, 5))
    df.boxplot(column='latency_us', by='event_type', grid=False, showfliers=False)
    plt.title(f'Network Operation Latency Distribution ({results_dir})')
    plt.suptitle('') # Remove auto-generated subtitle
    plt.ylabel('Latency (microseconds)')
    plt.xlabel('Operation Type')
    plt.tight_layout()
    plt.savefig(f'{results_dir}_net_latency.png')
    print(f"Saved {results_dir}_net_latency.png")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "native"
    target_dir = f"results/{mode}"
    
    print(f"Generating plots for {target_dir}...")
    plot_gpu_metrics(target_dir)
    plot_syscall_counts(target_dir)
    plot_network_latency(target_dir)