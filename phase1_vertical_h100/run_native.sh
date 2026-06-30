#!/usr/bin/env bash
#
# Phase 1 (vertical scaling, single-node multi-GPU) -- BARE-METAL run.
#
# Starts the full eBPF + NVML telemetry stack on the host, runs the ResNet DDP
# workload natively (no container), stops the probes, and exports a Perfetto
# trace. Sweeps both architectures by default:
#   resnet18 -> kernel-launch-bound (CPU starvation / GPU idle gaps)
#   resnet50 -> communication-bound (PCIe / AllReduce)
#
# Usage:
#   sudo ./run_native.sh                         # both arches, all GPUs
#   sudo GPUS=2 EPOCHS=5 ./run_native.sh
#   sudo ./run_native.sh --arch resnet50 --gpus 2 --epochs 5
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BPF="${REPO_ROOT}/src/bpf_kernel"
TEL="${REPO_ROOT}/src/telemetry"

# ---- Tunables (override via env or flags) ----
ARCHS="${ARCHS:-resnet18 resnet50}"
GPUS="${GPUS:-2}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-128}"
MASTER_PORT="${MASTER_PORT:-29501}"
DUR_CAP="${DUR_CAP:-3600}"          # profiler hard cap; we stop them when the workload ends
LLC_PERIOD="${LLC_PERIOD:-10000}"
RESULTS_BASE="${RESULTS_BASE:-${REPO_ROOT}/results/phase1}"

# ---- Python environments ----
# eBPF/BCC bindings live with the system python; the workload + NVML need torch/pynvml.
BCC_PY="${BCC_PY:-/usr/bin/python3}"
BCC_PYTHONPATH="${BCC_PYTHONPATH:-/usr/lib/python3/dist-packages}"
WORKLOAD_PY="${WORKLOAD_PY:-python3}"
TORCHRUN="${TORCHRUN:-torchrun}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCHS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --results) RESULTS_BASE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: eBPF profilers require root. Re-run with sudo." >&2
  exit 1
fi

run_bcc() { exec env PYTHONPATH="${BCC_PYTHONPATH}" "${BCC_PY}" "$@"; }
gpu_list() { seq -s, 0 $((GPUS - 1)); }

for ARCH in ${ARCHS}; do
  OUT="${RESULTS_BASE}/native_${ARCH}"
  mkdir -p "${OUT}"
  echo "============================================================"
  echo "  PHASE 1 NATIVE | arch=${ARCH} gpus=${GPUS} epochs=${EPOCHS}"
  echo "  -> ${OUT}"
  echo "============================================================"

  run_bcc "${BPF}/cpu_profiler.py"          --duration "${DUR_CAP}" --output "${OUT}/cpu.csv"        & P_CPU=$!
  run_bcc "${BPF}/syscall_counter.py"       --duration "${DUR_CAP}" --output "${OUT}/syscall.csv"    & P_SYS=$!
  run_bcc "${BPF}/net_profiler.py"          --duration "${DUR_CAP}" --output "${OUT}/net.csv"        & P_NET=$!
  run_bcc "${BPF}/llc_profiler.py"          --duration "${DUR_CAP}" --sample-period "${LLC_PERIOD}" --output "${OUT}/llc.csv" & P_LLC=$!
  run_bcc "${BPF}/cuda_uprobe_monitor.py"   --duration "${DUR_CAP}" --output "${OUT}/cuda_trace.csv" & P_CUDA=$!
  "${WORKLOAD_PY}" "${TEL}/nvml_monitor.py" --duration "${DUR_CAP}" --interval-ms 1 --gpus "$(gpu_list)" --output "${OUT}/nvml_gpu.csv" & P_NVML=$!

  echo "[*] Waiting 6s for eBPF programs to compile and attach..."
  sleep 6

  echo "[*] Launching workload..."
  WL_START=$(date +%s%N)
  if [[ "${GPUS}" -le 1 ]]; then
    "${WORKLOAD_PY}" "${SCRIPT_DIR}/resnet_ddp_workload.py" \
      --arch "${ARCH}" --gpus 1 --epochs "${EPOCHS}" --batch-size "${BATCH}" \
      --output "${OUT}/training.json"
  else
    "${TORCHRUN}" --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
      "${SCRIPT_DIR}/resnet_ddp_workload.py" \
      --arch "${ARCH}" --gpus "${GPUS}" --epochs "${EPOCHS}" --batch-size "${BATCH}" \
      --output "${OUT}/training.json"
  fi
  WL_MS=$(( ($(date +%s%N) - WL_START) / 1000000 ))
  echo "[*] Workload finished in ${WL_MS} ms. Stopping profilers..."

  for p in ${P_CPU} ${P_SYS} ${P_NET} ${P_LLC} ${P_CUDA} ${P_NVML}; do
    kill -INT "${p}" 2>/dev/null || true
  done
  wait ${P_CPU} ${P_SYS} ${P_NET} ${P_LLC} ${P_CUDA} ${P_NVML} 2>/dev/null || true

  echo "[*] Exporting Perfetto trace..."
  "${WORKLOAD_PY}" "${TEL}/perfetto_exporter.py" \
    --nvml "${OUT}/nvml_gpu.csv" --cuda "${OUT}/cuda_trace.csv" --net "${OUT}/net.csv" \
    --output "${OUT}/perfetto_trace.json" || true

  echo "[OK] arch=${ARCH} complete -> ${OUT}"
  echo
done

echo "All native Phase 1 runs complete under ${RESULTS_BASE}/"
