#!/usr/bin/env bash
#
# Quick post-provision sanity check for a profiled GPU node.
#   bash setup/verify_node.sh
#
echo "== kernel ==";        uname -r
echo "== nvidia-smi ==";    (nvidia-smi -L || echo "FAIL: nvidia-smi")
echo "== torch+cuda ==";    python3 -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())" 2>&1 | tail -1
echo "== pynvml ==";        python3 -c "import pynvml; pynvml.nvmlInit(); print('pynvml ok', pynvml.nvmlDeviceGetCount())" 2>&1 | tail -1
echo "== bcc ==";           /usr/bin/python3 -c "import bcc; print('bcc ok')" 2>&1 | tail -1
echo "== docker ==";        (docker --version || echo "FAIL: docker")
echo "== docker gpu ==";    (docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi -L 2>&1 | tail -3 || echo "FAIL: docker-gpu")
