# ebpf-gpu-profiler -- containerized ML workload image.
#
# Runs the ResNet DDP workload (Phase 1) or the FL client (Phase 2) inside a
# container. eBPF/NVML profiling happens on the HOST, so BCC is intentionally
# NOT installed here -- this image only carries the workload + CUDA runtime.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    iproute2 net-tools ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12.1 build of torch to match the base image.
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir numpy nvidia-ml-py fastapi "uvicorn[standard]" requests tqdm

WORKDIR /workspace

# Bring in the source tree (workloads + shared modules).
COPY src/ /workspace/src/
COPY phase1_vertical_h100/ /workspace/phase1_vertical_h100/
COPY phase2_horizontal_rtx3060/ /workspace/phase2_horizontal_rtx3060/

# Default: Phase 1 single-GPU smoke run. Override at `docker run`.
CMD ["python3", "phase1_vertical_h100/resnet_ddp_workload.py", "--arch", "resnet18", "--gpus", "1", "--epochs", "1"]
