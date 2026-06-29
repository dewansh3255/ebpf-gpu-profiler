#!/bin/bash

echo "==========================================="
echo "  Real Distributed Test (Sanskar's Laptop) "
echo "==========================================="

sudo -v
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

chmod +x monitor/monitor_network.py
chmod +x monitor/monitor_gpu.py

echo "[+] Starting eBPF Monitors in the background..."
sudo python3 monitor/monitor_network.py > /dev/null 2>&1 &
NET_PID=$!
sudo python3 monitor/monitor_gpu.py > /dev/null 2>&1 &
GPU_PID=$!

echo "[+] Waiting 8 seconds for BPF to compile..."
sleep 8

VENV_PYTHON="ebpf-env/bin/python3"

echo "[+] Starting Parameter Server (Port 8000)..."
$VENV_PYTHON src/server.py > /dev/null 2>&1 &
SERVER_PID=$!
sleep 3 

echo "[+] Starting Client 1..."
$VENV_PYTHON src/client.py 1 &
CLIENT1_PID=$!

echo "==========================================="
echo " KUNAL CAN NOW CONNECT TO CLIENT 2"
echo "==========================================="
echo ""
echo " ⚠️ DO NOT PRESS ENTER YET ⚠️"
echo " Wait for both your terminal AND Kunal's terminal to finish."

# THIS IS THE FIX: The script will completely pause here until you press Enter
read -p " Press [ENTER] only when both clients are 100% finished... "

echo "[!] Stopping monitors and server gracefully..."
sudo kill -SIGINT $NET_PID
sudo kill -SIGINT $GPU_PID
kill $SERVER_PID
sleep 5 # Give eBPF a moment to flush the CSVs to the hard drive

# Find and Graph
LATEST_NET_CSV=$(ls -t monitor/network_trace_PHASE1_*.csv 2>/dev/null | head -1)
LATEST_GPU_CSV=$(ls -t monitor/gpu_trace_PHASE2_*.csv 2>/dev/null | head -1)

if [[ -n "$LATEST_NET_CSV" && -n "$LATEST_GPU_CSV" ]]; then
    echo "[+] Found CSVs! Generating Unified Timeline Dashboard..."
    $VENV_PYTHON monitor/plot_unified_timeline.py "$LATEST_NET_CSV" "$LATEST_GPU_CSV"
    echo "[✓] SUCCESS! Open phase3_unified_timeline.png"
else
    echo "[!] Error: Could not find the generated CSV files."
fi