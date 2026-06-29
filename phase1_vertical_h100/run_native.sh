#!/bin/bash
# G_21_run_native.sh
# Group 21 - GRS Project Part A
#
# Orchestrates the full profiling pipeline in NATIVE (non-containerized) mode.
# Starts all eBPF profilers, runs the ML workload, then stops profilers
# and collects results.
#
# Usage:
#   sudo ./G_21_run_native.sh [--gpus N] [--epochs E] [--duration D]
#
# Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma

set -e

# ---- Configuration ----
GPUS="${GPUS:-1}"
EPOCHS="${EPOCHS:-5}"
PROFILE_DURATION="${PROFILE_DURATION:-120}"
RESULTS_DIR="${SCRIPT_DIR}/results/native"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_PORT="${MASTER_PORT:-29501}"

# ---- Resolve Python and torchrun from the venv ----
VENV_DIR="${SCRIPT_DIR}/venv"
if [[ -f "${VENV_DIR}/bin/python3" ]]; then
    PYTHON="${VENV_DIR}/bin/python3"
    TORCHRUN="${VENV_DIR}/bin/torchrun"
else
    PYTHON="$(which python3)"
    TORCHRUN="$(which torchrun 2>/dev/null || echo torchrun)"
fi

# ---- BCC requires system Python with PYTHONPATH (venv doesn't have bcc) ----
BCC_PYTHON="env PYTHONPATH=/usr/lib/python3/dist-packages /usr/bin/python3"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --duration) PROFILE_DURATION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  NATIVE PROFILING RUN"
echo "  GPUs: ${GPUS} | Epochs: ${EPOCHS} | Profile: ${PROFILE_DURATION}s"
echo "============================================================"

# ---- Check root ----
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: eBPF profilers require root privileges."
    echo "Run with: sudo ./G_21_run_native.sh"
    exit 1
fi

# ---- Create results directory ----
mkdir -p "${RESULTS_DIR}"

# ---- Start Profilers in Background ----
echo ""
echo "[1/5] Starting CPU profiler..."
${BCC_PYTHON} "${SCRIPT_DIR}/G_21_cpu_profiler.py" \
    --duration "${PROFILE_DURATION}" \
    --output "${RESULTS_DIR}/21_cpu_results.csv" &
PID_CPU=$!
echo "  PID: ${PID_CPU}"

echo "[2/5] Starting syscall counter..."
${BCC_PYTHON} "${SCRIPT_DIR}/G_21_syscall_counter.py" \
    --duration "${PROFILE_DURATION}" \
    --output "${RESULTS_DIR}/21_syscall_results.csv" &
PID_SYSCALL=$!
echo "  PID: ${PID_SYSCALL}"

echo "[3/5] Starting network profiler..."
${BCC_PYTHON} "${SCRIPT_DIR}/G_21_net_profiler.py" \
    --duration "${PROFILE_DURATION}" \
    --output "${RESULTS_DIR}/21_net_results.csv" &
PID_NET=$!
echo "  PID: ${PID_NET}"

echo "[4/5] Starting GPU monitor..."
${PYTHON} "${SCRIPT_DIR}/G_21_gpu_monitor_nvidia.py" \
    --duration "${PROFILE_DURATION}" \
    --interval 0.1 \
    --output "${RESULTS_DIR}/21_gpu_results.csv" &
PID_GPU=$!
echo "  PID: ${PID_GPU}"

# Give profilers time to attach
sleep 3

# ---- Run ML Workload ----
echo ""
echo "[5/5] Starting ML workload (native)..."
echo "============================================================"

WORKLOAD_START=$(date +%s%N)

if [ "${GPUS}" -eq 1 ]; then
    ${PYTHON} "${SCRIPT_DIR}/G_21_ml_workload.py" \
        --gpus 1 \
        --epochs "${EPOCHS}" \
        --output "${RESULTS_DIR}/21_training_native.json"
else
    ${TORCHRUN} --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
        "${SCRIPT_DIR}/G_21_ml_workload.py" \
        --gpus "${GPUS}" \
        --epochs "${EPOCHS}" \
        --output "${RESULTS_DIR}/21_training_native.json"
fi

WORKLOAD_END=$(date +%s%N)
WORKLOAD_DURATION=$(( (WORKLOAD_END - WORKLOAD_START) / 1000000 ))

echo ""
echo "============================================================"
echo "  ML workload completed in ${WORKLOAD_DURATION}ms"
echo "============================================================"

# ---- Stop Profilers ----
echo ""
echo "Stopping profilers..."
kill ${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU} 2>/dev/null || true
wait ${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU} 2>/dev/null || true
echo "All profilers stopped."

# ---- Summary ----
echo ""
echo "============================================================"
echo "  NATIVE RUN COMPLETE"
echo "============================================================"
echo ""
echo "Results saved in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
echo ""
echo "Workload wall time: ${WORKLOAD_DURATION}ms"
echo "============================================================"
