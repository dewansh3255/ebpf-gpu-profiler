#!/usr/bin/env bash
#
# Phase 2 -- FL PARAMETER SERVER launcher (runs on the unprofiled peer, e.g.
# siddhartha@192.168.3.133). No root / eBPF here; just serves global weights
# and runs FedAvg. Uses a local venv so no sudo is required.
#
# Usage:
#   ARCH=resnet18 CLIENTS=1 ./run_server.sh
#   ./run_server.sh --arch resnet50 --clients 1 --port 8000
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="${ARCH:-resnet18}"
CLIENTS="${CLIENTS:-1}"
PORT="${PORT:-8000}"
VENV="${VENV:-${SCRIPT_DIR}/.venv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --clients) CLIENTS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Bootstrap venv if missing (torch is assumed pre-installed system-wide on this
# node; we only add the lightweight server deps and reuse system torch).
PY="python3"
if [[ -x "${VENV}/bin/python3" ]]; then
  PY="${VENV}/bin/python3"
else
  echo "[*] Creating venv at ${VENV} (with system site-packages for torch)..."
  python3 -m venv --system-site-packages "${VENV}"
  PY="${VENV}/bin/python3"
  "${PY}" -m pip install --quiet --upgrade pip
  "${PY}" -m pip install --quiet fastapi "uvicorn[standard]" requests tqdm
fi

echo "============================================================"
echo "  PHASE 2 FL SERVER | arch=${ARCH} clients=${CLIENTS} port=${PORT}"
echo "  bind 0.0.0.0:${PORT}  (clients connect to this host's IP)"
echo "============================================================"

cd "${SCRIPT_DIR}"
FL_ARCH="${ARCH}" FL_EXPECTED_CLIENTS="${CLIENTS}" \
  "${PY}" fl_server.py --host 0.0.0.0 --port "${PORT}" --clients "${CLIENTS}"
