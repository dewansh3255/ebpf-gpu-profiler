#!/usr/bin/env python3
"""
nvml_monitor.py
High-resolution GPU telemetry collector built on NVML (via pynvml).

Replaces the previous nvidia-smi polling implementation. Instead of forking
`nvidia-smi` (tens of milliseconds per sample, subprocess + text parsing
overhead), this binds directly to the NVIDIA Management Library and samples
counters in-process. That lets us drive the sampling loop down to a ~1 ms
target interval and capture fine-grained behaviour that subprocess polling
would smear out.

Per sample, per GPU, it records:
  - GPU / memory controller utilisation (%)
  - Framebuffer memory used / total (MiB)
  - PCIe TX / RX throughput (MiB/s)   <- bus pressure during H2D/D2H + NCCL
  - Power draw (W)
  - Temperature (C)
  - SM / memory clocks (MHz)

Timestamps are captured on both CLOCK_MONOTONIC (nanoseconds, to align with
eBPF `bpf_ktime_get_ns()`) and the wall clock, so the resulting CSV can be
merged with the eBPF traces by the Perfetto exporter.

Usage:
    python3 nvml_monitor.py --duration 120 --interval-ms 1 \
        --output results/native/nvml_gpu.csv

Note on PCIe throughput resolution: NVML samples PCIe TX/RX over an internal
window (~20 ms on most drivers), so those two columns are coarser than the
1 ms loop. Utilisation, power, memory and clocks update at the sampling rate.
"""

import argparse
import csv
import os
import signal
import sys
import time

try:
    import pynvml
except ImportError:
    print("ERROR: pynvml is not installed. Install with: pip install nvidia-ml-py")
    sys.exit(1)


CLOCK_MONOTONIC = time.CLOCK_MONOTONIC

FIELDNAMES = [
    "ts_mono_ns",        # CLOCK_MONOTONIC ns  (aligns with eBPF ktime)
    "ts_unix_ns",        # wall-clock ns
    "gpu_index",
    "gpu_util_pct",
    "mem_util_pct",
    "mem_used_mib",
    "mem_total_mib",
    "pcie_tx_mibps",
    "pcie_rx_mibps",
    "power_w",
    "temperature_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
]


class NVMLMonitor:
    """In-process NVML sampler targeting a fixed inter-sample interval."""

    def __init__(self, duration, interval_ms, output_file, gpu_filter=None):
        self.duration = duration
        self.interval_s = interval_ms / 1000.0
        self.output_file = output_file
        self.gpu_filter = gpu_filter   # list[int] or None for all
        self.running = True
        self.samples = []

        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, *_):
        self.running = False

    # ---- NVML lifecycle ----
    def _init_devices(self):
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        indices = range(count) if self.gpu_filter is None else self.gpu_filter
        handles = []
        for i in indices:
            if i >= count:
                print(f"[NVML Monitor] WARNING: GPU index {i} not present "
                      f"(only {count} GPUs); skipping.")
                continue
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            print(f"[NVML Monitor] GPU {i}: {name}")
            handles.append((i, h))
        if not handles:
            raise RuntimeError("No usable GPUs found for monitoring.")
        return handles

    @staticmethod
    def _safe(fn, default=-1):
        """NVML calls can raise NVMLError for unsupported queries; degrade
        gracefully so one missing counter does not kill the whole sample."""
        try:
            return fn()
        except pynvml.NVMLError:
            return default

    def _sample_device(self, idx, handle, ts_mono, ts_unix):
        util = self._safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
        mem = self._safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
        power_mw = self._safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
        temp = self._safe(lambda: pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU))
        sm_clk = self._safe(lambda: pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_SM))
        mem_clk = self._safe(lambda: pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_MEM))
        # PCIe throughput is reported in KiB/s; convert to MiB/s.
        pcie_tx = self._safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
            handle, pynvml.NVML_PCIE_UTIL_TX_BYTES))
        pcie_rx = self._safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
            handle, pynvml.NVML_PCIE_UTIL_RX_BYTES))

        return {
            "ts_mono_ns": ts_mono,
            "ts_unix_ns": ts_unix,
            "gpu_index": idx,
            "gpu_util_pct": util.gpu if util != -1 else -1,
            "mem_util_pct": util.memory if util != -1 else -1,
            "mem_used_mib": round(mem.used / 1048576, 1) if mem != -1 else -1,
            "mem_total_mib": round(mem.total / 1048576, 1) if mem != -1 else -1,
            "pcie_tx_mibps": round(pcie_tx / 1024, 3) if pcie_tx != -1 else -1,
            "pcie_rx_mibps": round(pcie_rx / 1024, 3) if pcie_rx != -1 else -1,
            "power_w": round(power_mw / 1000.0, 2) if power_mw != -1 else -1,
            "temperature_c": temp,
            "sm_clock_mhz": sm_clk,
            "mem_clock_mhz": mem_clk,
        }

    def run(self):
        handles = self._init_devices()
        print(f"[NVML Monitor] Sampling every {self.interval_s * 1000:.1f} ms "
              f"for {self.duration}s ({len(handles)} GPU(s))...")

        start_mono = time.clock_gettime_ns(CLOCK_MONOTONIC)
        deadline_mono = start_mono + int(self.duration * 1e9)
        next_tick = time.perf_counter()

        n = 0
        while self.running:
            ts_mono = time.clock_gettime_ns(CLOCK_MONOTONIC)
            if ts_mono >= deadline_mono:
                break
            ts_unix = time.time_ns()

            for idx, handle in handles:
                self.samples.append(
                    self._sample_device(idx, handle, ts_mono, ts_unix))
            n += 1

            # Fixed-rate scheduling with drift compensation. If sampling fell
            # behind (NVML calls slower than the interval), skip the sleep and
            # resync rather than accumulating lag.
            next_tick += self.interval_s
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()

        elapsed = (time.clock_gettime_ns(CLOCK_MONOTONIC) - start_mono) / 1e9
        print(f"[NVML Monitor] Collected {n} ticks / {len(self.samples)} "
              f"samples in {elapsed:.2f}s "
              f"(effective {n / elapsed:.0f} Hz).")

        self._save()
        self._print_summary()
        pynvml.nvmlShutdown()

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)),
                    exist_ok=True)
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.samples)
        print(f"[NVML Monitor] Saved -> {self.output_file}")

    def _print_summary(self):
        if not self.samples:
            return
        by_gpu = {}
        for s in self.samples:
            by_gpu.setdefault(s["gpu_index"], []).append(s)
        print("\n" + "=" * 56)
        print("NVML SUMMARY")
        print("=" * 56)
        for idx in sorted(by_gpu):
            rows = by_gpu[idx]
            def avg(key):
                vals = [r[key] for r in rows if isinstance(r[key], (int, float)) and r[key] >= 0]
                return sum(vals) / len(vals) if vals else 0
            def mx(key):
                vals = [r[key] for r in rows if isinstance(r[key], (int, float)) and r[key] >= 0]
                return max(vals) if vals else 0
            print(f"\nGPU {idx}:")
            print(f"  Util      avg {avg('gpu_util_pct'):5.1f}%  max {mx('gpu_util_pct'):.0f}%")
            print(f"  Power     avg {avg('power_w'):6.1f}W  max {mx('power_w'):.0f}W")
            print(f"  Mem used  max {mx('mem_used_mib'):.0f} MiB")
            print(f"  PCIe TX   avg {avg('pcie_tx_mibps'):7.1f} MiB/s  max {mx('pcie_tx_mibps'):.0f}")
            print(f"  PCIe RX   avg {avg('pcie_rx_mibps'):7.1f} MiB/s  max {mx('pcie_rx_mibps'):.0f}")


def main():
    parser = argparse.ArgumentParser(description="NVML high-resolution GPU monitor")
    parser.add_argument("--duration", type=int, default=120,
                        help="Monitoring duration in seconds (default: 120)")
    parser.add_argument("--interval-ms", type=float, default=1.0,
                        help="Target sampling interval in milliseconds (default: 1.0)")
    parser.add_argument("--output", type=str, default="results/nvml_gpu.csv",
                        help="Output CSV path")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices to monitor (default: all)")
    args = parser.parse_args()

    gpu_filter = None
    if args.gpus:
        gpu_filter = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    NVMLMonitor(
        duration=args.duration,
        interval_ms=args.interval_ms,
        output_file=args.output,
        gpu_filter=gpu_filter,
    ).run()


if __name__ == "__main__":
    main()
