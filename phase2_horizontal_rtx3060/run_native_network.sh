#!/usr/bin/env bash
#
# Phase 2 (horizontal scaling, distributed FL) -- BARE-METAL profiled CLIENT.
#
# Runs on the profiled RTX 3060 node (iiitd@192.168.27.1). Starts the host
# eBPF + NVML telemetry, then runs the FL client which trains on its full
# CIFAR-10 shard for >=3 local epochs and exchanges weights with the remote
# parameter server -- saturating the TCP path so the network probe captures
# real communication-bound overhead. Sweeps resnet18 + resnet50.
#
# Usage:
#   sudo SERVER_URL=http://192.168.3.133:8000 ./run_native_network.sh
#   sudo ./run_native_network.sh --server-url http://192.168.3.133:8000 --arch resnet50
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BPF="${REPO_ROOT}/src/bpf_kernel"
TEL="${REPO_ROOT}/src/telemetry"

ARCHS="${ARCHS:-resnet18 resnet50}"
SERVER_URL="${SERVER_URL:-http://192.168.3.133:8000}"
CLIENT_ID="${CLIENT_ID:-1}"
NUM_CLIENTS="${NUM_CLIENTS:-1}"
ROUNDS="${ROUNDS:-3}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
DUR_CAP="${DUR_CAP:-3600}"
LLC_PERIOD="${LLC_PERIOD:-10000}"
RESULTS_BASE="${RESULTS_BASE:-${REPO_ROOT}/results/phase2}"

BCC_PY="${BCC_PY:-/usr/bin/python3}"
BCC_PYTHONPATH="${BCC_PYTHONPATH:-/usr/lib/python3/dist-packages}"
WORKLOAD_PY="${WORKLOAD_PY:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCHS="$2"; shift 2 ;;
    --server-url) SERVER_URL="$2"; shift 2 ;;
    --client-id) CLIENT_ID="$2"; shift 2 ;;
    --num-clients) NUM_CLIENTS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --local-epochs) LOCAL_EPOCHS="$2"; shift 2 ;;
    --results) RESULTS_BASE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: eBPF profilers require root. Re-run with sudo." >&2
  exit 1
fi

run_bcc() { exec env PYTHONPATH="${BCC_PYTHONPATH}" "${BCC_PY}" "$@"; }

echo "Server: ${SERVER_URL}  (make sure run_server.sh is already running there)"

for ARCH in ${ARCHS}; do
  OUT="${RESULTS_BASE}/native_${ARCH}"
  mkdir -p "${OUT}"
  echo "============================================================"
  echo "  PHASE 2 NATIVE CLIENT | arch=${ARCH} rounds=${ROUNDS} epochs=${LOCAL_EPOCHS}"
  echo "  -> ${OUT}"
  echo "============================================================"

  run_bcc "${BPF}/cpu_profiler.py"          --duration "${DUR_CAP}" --output "${OUT}/cpu.csv"        & P_CPU=$!
  run_bcc "${BPF}/syscall_counter.py"       --duration "${DUR_CAP}" --output "${OUT}/syscall.csv"    & P_SYS=$!
  run_bcc "${BPF}/net_profiler.py"          --duration "${DUR_CAP}" --output "${OUT}/net.csv"        & P_NET=$!
  run_bcc "${BPF}/llc_profiler.py"          --duration "${DUR_CAP}" --sample-period "${LLC_PERIOD}" --output "${OUT}/llc.csv" & P_LLC=$!
  run_bcc "${BPF}/cuda_uprobe_monitor.py"   --duration "${DUR_CAP}" --output "${OUT}/cuda_trace.csv" & P_CUDA=$!
  "${WORKLOAD_PY}" "${TEL}/nvml_monitor.py" --duration "${DUR_CAP}" --interval-ms 1 --gpus 0 --output "${OUT}/nvml_gpu.csv" & P_NVML=$!

  echo "[*] Waiting 6s for eBPF programs to attach..."
  sleep 6

  echo "[*] Launching FL client..."
  WL_START=$(date +%s%N)
  "${WORKLOAD_PY}" "${SCRIPT_DIR}/fl_client.py" \
    --client-id "${CLIENT_ID}" --num-clients "${NUM_CLIENTS}" \
    --server-url "${SERVER_URL}" --arch "${ARCH}" \
    --rounds "${ROUNDS}" --local-epochs "${LOCAL_EPOCHS}" \
    2>&1 | tee "${OUT}/client_log.txt"
  WL_MS=$(( ($(date +%s%N) - WL_START) / 1000000 ))
  echo "[*] Client finished in ${WL_MS} ms. Stopping profilers..."

  for p in ${P_CPU} ${P_SYS} ${P_NET} ${P_LLC} ${P_CUDA} ${P_NVML}; do
    kill -INT "${p}" 2>/dev/null || true
  done
  wait ${P_CPU} ${P_SYS} ${P_NET} ${P_LLC} ${P_CUDA} ${P_NVML} 2>/dev/null || true

  echo "[*] Exporting Perfetto trace..."
  "${WORKLOAD_PY}" "${TEL}/perfetto_exporter.py" \
    --nvml "${OUT}/nvml_gpu.csv" --cuda "${OUT}/cuda_trace.csv" --net "${OUT}/net.csv" \
    --output "${OUT}/perfetto_trace.json" || true

  echo "[OK] arch=${ARCH} complete -> ${OUT}"
  echo "[!] Restart run_server.sh on the server before the next arch (server advances rounds per arch)."
  echo
done

echo "All native Phase 2 client runs complete under ${RESULTS_BASE}/"
