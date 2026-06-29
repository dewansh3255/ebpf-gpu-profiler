#!/usr/bin/env bash
#
# Provision a Linux GPU host as a PROFILED benchmarking node.
#
# Installs everything needed to run a profiled workload on this machine:
#   - NVIDIA driver (via ubuntu-drivers; needs a reboot afterwards)
#   - eBPF / BCC toolchain + matching kernel headers
#   - PyTorch (CUDA 12.1 build) + telemetry deps into the system Python
#   - Docker + NVIDIA Container Toolkit (for the Docker-vs-bare-metal runs)
#
# Target: Ubuntu 22.04, run as root:
#   sudo bash setup/provision_profiled_node.sh
#
# After it finishes: REBOOT, then verify with setup/verify_node.sh
#
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

# ---- base packages: build tools, eBPF/BCC, kernel headers, docker ----
apt-get update
apt-get install -y \
  build-essential python3-pip python3-dev \
  "linux-headers-$(uname -r)" bpfcc-tools python3-bpfcc libbpfcc-dev \
  docker.io curl ca-certificates gnupg

# ---- NVIDIA driver (skip if a working driver is already present) ----
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "[provision] NVIDIA driver already functional; skipping driver install."
else
  echo "[provision] Installing recommended NVIDIA driver via ubuntu-drivers..."
  apt-get install -y ubuntu-drivers-common
  ubuntu-drivers autoinstall
fi

# ---- PyTorch + telemetry deps (system python; matches host BCC python) ----
PIP_FLAGS=""
python3 -m pip install --upgrade pip || PIP_FLAGS="--break-system-packages"
python3 -m pip install ${PIP_FLAGS} torch torchvision --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install ${PIP_FLAGS} nvidia-ml-py numpy requests tqdm

# ---- Docker + NVIDIA Container Toolkit ----
install -m 0755 -d /usr/share/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker || true

echo "PROVISION_DONE -- REBOOT required to load the NVIDIA kernel module."
