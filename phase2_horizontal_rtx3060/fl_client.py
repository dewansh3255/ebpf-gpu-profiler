#!/usr/bin/env python3
"""
fl_client.py
Federated-learning network client.

Each client trains on its full CIFAR-10 shard for several local epochs per
round, then uploads compressed weights to the parameter server. Training on the
full shard (not a 100-image debug slice) for >=3 local epochs produces real,
sizeable gradient payloads -- which is the whole point of Phase 2: saturating
the TCP path between nodes so the eBPF network/scheduler probes have something
meaningful to measure.

Usage:
    python3 fl_client.py --client-id 1 --num-clients 2 \
        --server-url http://192.168.27.1:8000 --arch resnet18 \
        --rounds 3 --local-epochs 3
"""

import argparse
import io
import os
import time
import zlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # first dedicated GPU

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from fl_model import build_model


class FederatedClient:
    def __init__(self, client_id, dataset, arch="resnet18", batch_size=64, lr=0.01):
        self.client_id = client_id
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=2, pin_memory=True)
        self.model = build_model(arch, num_classes=10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                   momentum=0.9, weight_decay=5e-4)

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def train_local_model(self, epochs=3):
        self.model.train()
        print(f"--- Client {self.client_id}: {epochs} local epoch(s) "
              f"on {self.device} ({len(self.dataset)} samples) ---")
        for epoch in range(epochs):
            running_loss = 0.0
            bar = tqdm(self.dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}")
            for inputs, labels in bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                bar.set_postfix({"loss": f"{loss.item():.4f}"})
            print(f"Client {self.client_id} | Epoch {epoch+1} | "
                  f"Avg Loss: {running_loss / len(self.dataloader):.4f}")


def run_network_client(client_id, server_url, num_clients=2, arch="resnet18",
                       rounds=3, local_epochs=3):
    from fl_dataset import get_cifar10_datasets

    print(f"Client {client_id} initialising data (arch={arch})...")
    client_datasets, _ = get_cifar10_datasets(num_clients=num_clients)

    # Train on the FULL shard for this client (no debug Subset slice).
    my_data = client_datasets[client_id - 1]
    client = FederatedClient(client_id=client_id, dataset=my_data, arch=arch)

    for r in range(1, rounds + 1):
        print(f"\n=== ROUND {r} ===")

        # Synchronisation barrier: wait until the server advances to round r.
        print(f"Waiting for server to reach round {r}...")
        while True:
            try:
                status = requests.get(f"{server_url}/status", timeout=10).json()
                if status["current_round"] == r:
                    print(f"Server ready for round {r}.")
                    break
                time.sleep(2)
            except Exception as e:
                print("Error connecting to server:", e)
                time.sleep(2)

        # 1. Download + decompress global weights.
        print("Downloading global weights...")
        try:
            response = requests.get(f"{server_url}/get_weights", timeout=120)
            if response.status_code != 200:
                print("Failed to reach server. Exiting.")
                break
            buffer = io.BytesIO(zlib.decompress(response.content))
            global_weights = torch.load(buffer, map_location="cpu", weights_only=True)
            client.set_weights(global_weights)
        except Exception as e:
            print(f"Connection error: {e}")
            break

        # 2. Train locally on the full shard.
        client.train_local_model(epochs=local_epochs)

        # 3. Compress + upload updated weights.
        print("Compressing and uploading trained weights...")
        buffer = io.BytesIO()
        torch.save(client.get_weights(), buffer)
        compressed = zlib.compress(buffer.getvalue(), level=3)
        files = {"file": ("weights.pth", io.BytesIO(compressed),
                          "application/octet-stream")}
        requests.post(f"{server_url}/upload_weights", files=files, timeout=120)
        print(f"Round {r} upload complete.")


def main():
    parser = argparse.ArgumentParser(description="FL network client")
    parser.add_argument("--client-id", type=int,
                        default=int(os.environ.get("FL_CLIENT_ID", "1")))
    parser.add_argument("--num-clients", type=int,
                        default=int(os.environ.get("FL_NUM_CLIENTS", "2")))
    parser.add_argument("--server-url", type=str,
                        default=os.environ.get("FL_SERVER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--arch", type=str,
                        default=os.environ.get("FL_ARCH", "resnet18"),
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=3,
                        help="Local epochs per round (>=3 to saturate the network)")
    args = parser.parse_args()

    run_network_client(
        client_id=args.client_id,
        server_url=args.server_url,
        num_clients=args.num_clients,
        arch=args.arch,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
    )


if __name__ == "__main__":
    main()
