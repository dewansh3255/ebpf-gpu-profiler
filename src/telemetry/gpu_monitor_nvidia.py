#!/usr/bin/env python3
"""
21_gpu_monitor.py
Group 21 - GRS Project Part A

GPU monitoring tool that collects GPU utilization, memory usage,
temperature, power draw, and execution timeline during workload runs.
Initial implementation uses nvidia-smi polling; eGPU-based deep
instrumentation is planned for Part B.

Usage:
    python3 21_gpu_monitor.py --duration 60 --interval 0.5 --output 21_gpu_results.csv

Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime


class GPUMonitor:
    """
    Polls nvidia-smi to collect GPU metrics at regular intervals.
    This is the initial implementation for Part A. Part B will integrate
    eGPU-based PTX injection for deeper GPU kernel-level monitoring.
    """

    # nvidia-smi query fields
    QUERY_FIELDS = [
        "index",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.total",
        "memory.used",
        "memory.free",
        "temperature.gpu",
        "power.draw",
        "clocks.current.sm",
        "clocks.current.memory",
        "pstate",
    ]

    FIELD_NAMES = [
        "gpu_index", "gpu_name", "gpu_util_pct", "mem_util_pct",
        "mem_total_mib", "mem_used_mib", "mem_free_mib",
        "temperature_c", "power_w", "sm_clock_mhz",
        "mem_clock_mhz", "pstate",
    ]

    def __init__(self, duration, interval, output_file):
        self.duration = duration
        self.interval = interval
        self.output_file = output_file
        self.running = True
        self.samples = []

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.running = False

    def _check_nvidia_smi(self):
        """Verify nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpus = result.stdout.strip().split("\n")
                print(f"[GPU Monitor] Detected {len(gpus)} GPU(s):")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.strip()}")
                return True
            else:
                print(f"[GPU Monitor] nvidia-smi error: {result.stderr}")
                return False
        except FileNotFoundError:
            print("[GPU Monitor] ERROR: nvidia-smi not found.")
            print("  Make sure NVIDIA drivers are installed.")
            return False
        except subprocess.TimeoutExpired:
            print("[GPU Monitor] ERROR: nvidia-smi timed out.")
            return False

    def _query_gpu(self):
        """Query GPU metrics via nvidia-smi."""
        query = ",".join(self.QUERY_FIELDS)
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 f"--query-gpu={query}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return None

            samples = []
            timestamp = time.time()
            for line in result.stdout.strip().split("\n"):
                values = [v.strip() for v in line.split(",")]
                if len(values) >= len(self.FIELD_NAMES):
                    sample = {"timestamp": timestamp}
                    for field, val in zip(self.FIELD_NAMES, values):
                        # Try to convert numeric fields
                        try:
                            if "." in val:
                                sample[field] = float(val)
                            else:
                                sample[field] = int(val)
                        except (ValueError, TypeError):
                            sample[field] = val
                    samples.append(sample)
            return samples
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"[GPU Monitor] Query error: {e}")
            return None

    def run(self):
        """Main monitoring loop."""
        print(f"[GPU Monitor] Starting GPU monitoring...")

        if not self._check_nvidia_smi():
            print("[GPU Monitor] Cannot proceed without nvidia-smi.")
            print("[GPU Monitor] Generating sample data for testing.")
            self._generate_sample_data()
            return

        print(f"[GPU Monitor] Sampling every {self.interval}s "
              f"for {self.duration}s...")
        start = time.time()
        sample_count = 0

        while self.running and (time.time() - start) < self.duration:
            samples = self._query_gpu()
            if samples:
                self.samples.extend(samples)
                sample_count += 1

            time.sleep(self.interval)

        elapsed = time.time() - start
        print(f"\n[GPU Monitor] Stopped after {elapsed:.1f}s "
              f"({sample_count} samples)")

        self._print_summary()
        self._save_results()

    def _generate_sample_data(self):
        """Generate sample data for testing when no GPU is available."""
        print("[GPU Monitor] Generating 60 seconds of sample data...")
        import random

        for i in range(120):  # 0.5s intervals for 60s
            self.samples.append({
                "timestamp": time.time() + i * 0.5,
                "gpu_index": 0,
                "gpu_name": "SAMPLE-GPU (no real GPU detected)",
                "gpu_util_pct": random.uniform(40, 95),
                "mem_util_pct": random.uniform(30, 80),
                "mem_total_mib": 16384,
                "mem_used_mib": random.uniform(4000, 12000),
                "mem_free_mib": random.uniform(4000, 12000),
                "temperature_c": random.uniform(45, 75),
                "power_w": random.uniform(100, 250),
                "sm_clock_mhz": 1500,
                "mem_clock_mhz": 5001,
                "pstate": "P0",
            })

        self._print_summary()
        self._save_results()

    def _print_summary(self):
        if not self.samples:
            print("[GPU Monitor] No samples collected.")
            return

        print(f"\n{'=' * 60}")
        print("GPU MONITORING SUMMARY")
        print(f"{'=' * 60}")

        # Group by GPU
        gpus = {}
        for s in self.samples:
            idx = s.get("gpu_index", 0)
            if idx not in gpus:
                gpus[idx] = []
            gpus[idx].append(s)

        for gpu_idx, gpu_samples in sorted(gpus.items()):
            print(f"\nGPU {gpu_idx}: {gpu_samples[0].get('gpu_name', 'N/A')}")

            # Compute stats for numeric fields
            for field in ["gpu_util_pct", "mem_util_pct", "mem_used_mib",
                          "temperature_c", "power_w"]:
                values = []
                for s in gpu_samples:
                    v = s.get(field)
                    if isinstance(v, (int, float)):
                        values.append(v)
                if values:
                    unit = {"gpu_util_pct": "%", "mem_util_pct": "%",
                            "mem_used_mib": "MiB", "temperature_c": "°C",
                            "power_w": "W"}.get(field, "")
                    label = field.replace("_", " ").title()
                    print(f"  {label}:")
                    print(f"    Avg: {sum(values) / len(values):.1f} {unit}")
                    print(f"    Min: {min(values):.1f} {unit}")
                    print(f"    Max: {max(values):.1f} {unit}")

        print(f"\nTotal samples: {len(self.samples)}")

    def _save_results(self):
        if not self.samples:
            return

        fieldnames = ["timestamp"] + self.FIELD_NAMES
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)

        print(f"\n[GPU Monitor] Results saved to: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="GPU Monitor (nvidia-smi based) - Group 21"
    )
    parser.add_argument("--duration", type=int, default=60,
                        help="Monitoring duration in seconds (default: 60)")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="Sampling interval in seconds (default: 0.1)")
    parser.add_argument("--output", type=str,
                        default="21_gpu_results.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    monitor = GPUMonitor(
        duration=args.duration,
        interval=args.interval,
        output_file=args.output,
    )
    monitor.run()


if __name__ == "__main__":
    main()
