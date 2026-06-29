#!/bin/bash
# G_21_run_container.sh
# Group 21 - GRS Project Part A
#
# Orchestrates the full profiling pipeline in CONTAINERIZED mode.
# Starts eBPF profilers on the HOST (they need kernel access), then
# runs the ML workload INSIDE a Docker container. This setup mirrors
# real production deployments where the container runs the application
# and the host monitors it.
#
# Usage:
#   sudo ./G_21_run_container.sh [--gpus N] [--epochs E] [--duration D]
#
# Prerequisites:
#   - Docker image built: ./G_21_container_setup.sh build
#
# Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma

set -e

# ---- Configuration ----
GPUS="${GPUS:-1}"
EPOCHS="${EPOCHS:-5}"
PROFILE_DURATION="${PROFILE_DURATION:-120}"
RESULTS_DIR="${SCRIPT_DIR}/results/container"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="group21-ml-profiling"
CONTAINER_NAME="group21-profiled-run"
MASTER_PORT="${MASTER_PORT:-29501}"

# ---- Resolve Python from the venv ----
VENV_DIR="${SCRIPT_DIR}/venv"
if [[ -f "${VENV_DIR}/bin/python3" ]]; then
    PYTHON="${VENV_DIR}/bin/python3"
else
    PYTHON="$(which python3)"
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
echo "  CONTAINERIZED PROFILING RUN"
echo "  GPUs: ${GPUS} | Epochs: ${EPOCHS} | Profile: ${PROFILE_DURATION}s"
echo "============================================================"

# ---- Check root ----
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: eBPF profilers require root privileges."
    echo "Run with: sudo ./G_21_run_container.sh"
    exit 1
fi

# ---- Check Docker image exists ----
if ! docker image inspect "${IMAGE_NAME}" &>/dev/null; then
    echo "ERROR: Docker image '${IMAGE_NAME}' not found."
    echo "Build it first: ./G_21_container_setup.sh build"
    exit 1
fi

# ---- Create results directory ----
mkdir -p "${RESULTS_DIR}"

# ---- Clean up any existing container ----
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ---- Start Profilers on HOST (they need kernel access) ----
echo ""
echo "[1/5] Starting CPU profiler (host)..."
${BCC_PYTHON} "${SCRIPT_DIR}/G_21_cpu_profiler.py" \
    --duration "${PROFILE_DURATION}" \
    --output "${RESULTS_DIR}/21_cpu_results.csv" &
PID_CPU=$!

echo "[2/5] Starting syscall counter (host)..."
${BCC_PYTHON} "${SCRIPT_DIR}/G_21_syscall_counter.py" \
    --duration "${PROFILE_DURATION}" \
    --output "${RESULTS_DIR}/21_syscall_results.csv" &
PID_SYSCALL=$!

echo "[3/5] Starting network profiler (host)..."
${BCC_PYTHON} "${SCRIPT_DIR}/G_21_net_profiler.py" \
    --duration "${PROFILE_DURATION}" \
    --output "${RESULTS_DIR}/21_net_results.csv" &
PID_NET=$!

echo "[4/5] Starting GPU monitor (host)..."
${PYTHON} "${SCRIPT_DIR}/G_21_gpu_monitor_nvidia.py" \
    --duration "${PROFILE_DURATION}" \
    --interval 0.1 \
    --output "${RESULTS_DIR}/21_gpu_results.csv" &
PID_GPU=$!

# Give profilers time to attach
sleep 3

# ---- Run ML Workload INSIDE Container ----
echo ""
echo "[5/5] Starting ML workload (containerized)..."
echo "============================================================"

WORKLOAD_START=$(date +%s%N)

if [ "${GPUS}" -eq 1 ]; then
    docker run \
        --name "${CONTAINER_NAME}" \
        --gpus all \
        --shm-size=2g \
        --ulimit memlock=-1 \
        --network=bridge \
        -v "${SCRIPT_DIR}/results:/workspace/results" \
        -v "${SCRIPT_DIR}/data:/workspace/data" \
        "${IMAGE_NAME}" \
        python3 G_21_ml_workload.py \
            --gpus 1 \
            --epochs "${EPOCHS}" \
            --output "results/container/21_training_container.json"
else
    docker run \
        --name "${CONTAINER_NAME}" \
        --gpus all \
        --shm-size=2g \
        --ulimit memlock=-1 \
        --network=bridge \
        -v "${SCRIPT_DIR}/results:/workspace/results" \
        -v "${SCRIPT_DIR}/data:/workspace/data" \
        "${IMAGE_NAME}" \
        bash -c "torchrun --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} G_21_ml_workload.py \
            --gpus ${GPUS} \
            --epochs ${EPOCHS} \
            --output results/container/21_training_container.json"
fi

WORKLOAD_END=$(date +%s%N)
WORKLOAD_DURATION=$(( (WORKLOAD_END - WORKLOAD_START) / 1000000 ))

echo ""
echo "============================================================"
echo "  Containerized workload completed in ${WORKLOAD_DURATION}ms"
echo "============================================================"

# ---- Stop Profilers ----
echo ""
echo "Stopping profilers..."
kill ${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU} 2>/dev/null || true
wait ${PID_CPU} ${PID_SYSCALL} ${PID_NET} ${PID_GPU} 2>/dev/null || true

# ---- Fix permissions on container-written result files ----
chmod -R a+rw "${RESULTS_DIR}/" 2>/dev/null || true

# ---- Capture Container Stats ----
echo ""
echo "Capturing container metadata..."
docker inspect "${CONTAINER_NAME}" > "${RESULTS_DIR}/21_container_inspect.json" 2>/dev/null || true

# ---- Summary ----
echo ""
echo "============================================================"
echo "  CONTAINERIZED RUN COMPLETE"
echo "============================================================"
echo ""
echo "Results saved in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
echo ""
echo "Workload wall time: ${WORKLOAD_DURATION}ms"
echo "============================================================"

# Cleanup container
docker rm "${CONTAINER_NAME}" 2>/dev/null || true
