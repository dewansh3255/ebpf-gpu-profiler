# ebpf-gpu-profiler

A bare-metal vs. container benchmarking suite that quantifies the OS-level
overhead Docker imposes on GPU machine-learning workloads, using **eBPF**
kernel instrumentation correlated with **NVML** GPU telemetry.

Most "Docker has near-zero overhead" claims look only at wall-clock throughput.
This suite goes a layer deeper: it attaches eBPF probes to the scheduler, the
syscall layer, the network stack, the CPU PMU, and the CUDA user-space driver,
then lines that up with millisecond-resolution GPU counters. That exposes costs
that wall-clock timing hides — kernel-launch-bound GPU starvation, container
network-path latency, and namespace-induced cache pollution.

---

## Two-phase architecture

The suite is organized around two complementary scaling axes:

### Phase 1 — Vertical scaling (single node, multi-GPU)
`phase1_vertical_h100/` — a ResNet DDP workload on a single multi-GPU node
(developed for 2× H100). Profiles intra-node behaviour: NCCL all-reduce,
inter-GPU PCIe traffic, and per-step kernel-launch overhead, bare-metal vs.
inside a container.

### Phase 2 — Horizontal scaling (distributed federated learning)
`phase2_horizontal_rtx3060/` — a federated-learning setup where a profiled
RTX 3060 client trains on its full CIFAR-10 shard and exchanges weights with a
remote parameter server. Profiles the **communication-bound** regime: the TCP
path, scheduler latency, and syscall overhead of the container network stack
vs. the host.

### Dual-model design
Both phases accept `--arch resnet18 | resnet50`:
- **ResNet-18** — light kernels; the GPU drains its queue fast, so the limiter
  is how quickly the CPU issues the next kernel. Exposes **kernel-launch-bound
  CPU starvation** (GPU idle gaps).
- **ResNet-50** — deep, bottleneck-heavy; large activations and ~98 MB gradient
  tensors per update. Exposes **communication-bound** PCIe / network
  synchronisation overhead.

---

## What gets measured

| Layer | Tool | Signal |
|---|---|---|
| Scheduler | `src/bpf_kernel/cpu_profiler.py` | context switches, run-queue latency |
| Syscalls | `src/bpf_kernel/syscall_counter.py` | per-syscall count + latency |
| Network | `src/bpf_kernel/net_profiler.py` | TCP send/recv bytes + latency |
| LLC cache | `src/bpf_kernel/llc_profiler.py` | last-level-cache misses via CPU PMU; per-cgroup attribution |
| CUDA driver | `src/bpf_kernel/cuda_uprobe_monitor.py` | `cuLaunchKernel` queue overhead **vs.** `cuStreamSynchronize` hardware-exec+sync time |
| GPU device | `src/telemetry/nvml_monitor.py` | utilisation, memory, **PCIe TX/RX**, power (NVML, ~1 ms loop) |
| Unified view | `src/telemetry/perfetto_exporter.py` | merges everything into a Chrome Trace Event JSON for [ui.perfetto.dev](https://ui.perfetto.dev) |

### The "true execution probe"
`cuda_uprobe_monitor.py` separates two costs a launch-only trace conflates:
`cuLaunchKernel` only *enqueues* work (CPU-side driver overhead), while
`cuStreamSynchronize` *blocks* until the device finishes (real hardware
execution + sync). The analysis layer then derives **GPU idle gaps** — the time
between a sync returning and the next kernel launch — which is the direct
measure of CPU-side GPU starvation.

---

## Repository layout

```
ebpf-gpu-profiler/
├── src/
│   ├── bpf_kernel/        # eBPF/BCC profilers (cpu, net, syscall, llc, cuda uprobe)
│   └── telemetry/         # NVML monitor + Perfetto exporter
├── phase1_vertical_h100/  # ResNet DDP workload + run_native.sh / run_docker.sh
├── phase2_horizontal_rtx3060/
│   ├── fl_server.py fl_client.py fl_dataset.py fl_model.py fl_main.py
│   ├── run_server.sh              # FL parameter server (unprofiled peer)
│   ├── run_native_network.sh      # profiled bare-metal client
│   └── run_docker_network.sh      # profiled containerized client
├── analysis_and_plots/    # gpu_idle_gaps.py, phase2_compare.py, plotting/aggregation
├── setup/                 # provision_profiled_node.sh, verify_node.sh
├── Dockerfile             # containerized workload image
└── requirements.txt
```

---

## Setup

eBPF profilers need root and run under the system Python that carries the BCC
bindings; the workload + NVML need PyTorch (CUDA 12.1) and `nvidia-ml-py`.

Provision a profiled GPU node (Ubuntu 22.04) in one shot:

```bash
sudo bash setup/provision_profiled_node.sh   # driver, BCC, torch, docker, toolkit
# reboot to load the NVIDIA kernel module, then:
bash setup/verify_node.sh
```

The FL parameter-server peer only needs PyTorch (no root): `run_server.sh`
installs its light deps into the user site automatically.

> BCC is **not** a pip package — it is installed from the distro
> (`bpfcc-tools python3-bpfcc`). The eBPF scripts run as:
> `sudo env PYTHONPATH=/usr/lib/python3/dist-packages python3 <script>.py`

---

## Running

### Phase 1 (single-node multi-GPU)
```bash
# bare metal (sweeps resnet18 + resnet50, all GPUs)
sudo ./phase1_vertical_h100/run_native.sh
# docker (build the image first)
docker build -t ebpf-gpu-profiler:latest .
sudo ./phase1_vertical_h100/run_docker.sh
```

### Phase 2 (distributed FL)
```bash
# on the server peer:
ARCH=resnet18 CLIENTS=1 ./phase2_horizontal_rtx3060/run_server.sh
# on the profiled RTX 3060 client (bare metal, then docker):
sudo SERVER_URL=http://<server-ip>:8100 ./phase2_horizontal_rtx3060/run_native_network.sh
sudo SERVER_URL=http://<server-ip>:8100 ./phase2_horizontal_rtx3060/run_docker_network.sh
```

Each run writes to `results/<phase>/<mode>_<arch>/`: the eBPF CSVs, `nvml_gpu.csv`,
and a merged `perfetto_trace.json`. Profilers are memory-bounded (`--max-events`)
and stop cleanly when the workload finishes.

---

## Analysis

```bash
# GPU idle gaps (CPU starvation), native vs docker
python3 analysis_and_plots/gpu_idle_gaps.py \
    --cuda results/phase2/native_resnet18/cuda_trace.csv \
    --cuda2 results/phase2/docker_resnet18/cuda_trace.csv \
    --label1 Native --label2 Docker --plot results/phase2/plots/idle.png

# full bare-metal vs docker comparison (both arches)
python3 analysis_and_plots/phase2_compare.py \
    --results-base results/phase2 --archs resnet18 resnet50 \
    --out results/phase2/plots

# view a unified timeline
#   open results/phase2/<run>/perfetto_trace.json at https://ui.perfetto.dev
```

---

## Requirements

PyTorch + torchvision (CUDA 12.1), `nvidia-ml-py`, numpy, matplotlib, FastAPI +
uvicorn + `python-multipart` (server), requests, tqdm. See `requirements.txt`.
BCC and kernel headers come from the distro (see Setup).
