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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --clients) CLIENTS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Server deps (torch is assumed pre-installed system-wide on this node). This
# node may lack `python3-venv` and sudo, so we install the lightweight server
# deps into the user site with pip --user instead of a virtualenv.
PY="${PY:-python3}"
echo "[*] Ensuring server deps (fastapi/uvicorn/requests/tqdm) in user site..."
"${PY}" -m pip install --user --quiet fastapi uvicorn requests tqdm || true

echo "============================================================"
echo "  PHASE 2 FL SERVER | arch=${ARCH} clients=${CLIENTS} port=${PORT}"
echo "  bind 0.0.0.0:${PORT}  (clients connect to this host's IP)"
echo "============================================================"

cd "${SCRIPT_DIR}"
FL_ARCH="${ARCH}" FL_EXPECTED_CLIENTS="${CLIENTS}" \
  "${PY}" fl_server.py --host 0.0.0.0 --port "${PORT}" --clients "${CLIENTS}"
