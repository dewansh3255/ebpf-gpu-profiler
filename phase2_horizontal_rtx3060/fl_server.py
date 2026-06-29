#!/usr/bin/env python3
"""
fl_server.py
Federated-learning global parameter server (FastAPI + FedAvg).

Clients pull the current global weights, train locally on their CIFAR-10 shard,
and upload updated weights; the server averages them (FedAvg) once all expected
clients report in, then evaluates the new global model on the held-out test set.

The model architecture is selected via the FL_ARCH environment variable
(resnet18 | resnet50) so it can be set before uvicorn imports this module:

    FL_ARCH=resnet50 python3 fl_server.py --clients 2 --port 8000

The FederatedServer class also supports an in-process simulation API
(get_global_weights / aggregate_weights(list)) used by fl_main.py.
"""

import argparse
import io
import os
import zlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # first dedicated GPU

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from torch.utils.data import DataLoader

from fl_model import build_model

ARCH = os.environ.get("FL_ARCH", "resnet18")
EXPECTED_CLIENTS = int(os.environ.get("FL_EXPECTED_CLIENTS", "2"))

app = FastAPI(title="Federated Learning Global Server")


class FederatedServer:
    def __init__(self, arch=ARCH, expected_clients=EXPECTED_CLIENTS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch
        self.global_model = build_model(arch, num_classes=10).to(self.device)

        self.expected_clients = expected_clients
        self.received_weights = []
        self.current_round = 1

        print(f"[Server] arch={arch}  expected_clients={expected_clients}  "
              f"device={self.device}")
        print("[Server] Loading global test set for evaluation...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR10(root="./data", train=False, download=True,
                                   transform=transform)
        self.test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    # ---- simulation helper (used by fl_main.py) ----
    def get_global_weights(self):
        return {k: v.cpu() for k, v in self.global_model.state_dict().items()}

    def evaluate_global_model(self):
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"\n[EVALUATION] Round {self.current_round - 1} "
              f"global accuracy: {accuracy:.2f}%")
        print("-" * 50)
        return accuracy

    def aggregate_weights(self, client_weights=None):
        """FedAvg. If client_weights is given (simulation), average those;
        otherwise average the weights collected over the network."""
        weights = client_weights if client_weights is not None else self.received_weights
        if not weights:
            print("[Server] No client weights to aggregate; skipping.")
            return
        print(f"\n[Server] Aggregating round {self.current_round} "
              f"from {len(weights)} clients...")
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            acc = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for client_dict in weights:
                acc += client_dict[key].to(self.device, dtype=torch.float32)
            global_dict[key] = (acc / len(weights)).to(global_dict[key].dtype)
        self.global_model.load_state_dict(global_dict)

        self.current_round += 1
        self.received_weights = []
        self.evaluate_global_model()


fl_server = FederatedServer()


@app.get("/status")
async def get_status():
    return {"current_round": fl_server.current_round, "arch": fl_server.arch}


@app.get("/get_weights")
async def get_weights():
    buffer = io.BytesIO()
    torch.save(fl_server.global_model.state_dict(), buffer)
    compressed = zlib.compress(buffer.getvalue(), level=3)
    return Response(content=compressed, media_type="application/octet-stream")


@app.post("/upload_weights")
async def upload_weights(file: UploadFile = File(...)):
    contents = await file.read()
    decompressed = zlib.decompress(contents)
    buffer = io.BytesIO(decompressed)
    client_state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    fl_server.received_weights.append(client_state_dict)
    print(f"[Network] Received weights "
          f"({len(fl_server.received_weights)}/{fl_server.expected_clients})")
    if len(fl_server.received_weights) >= fl_server.expected_clients:
        fl_server.aggregate_weights()
    return {"status": "success"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL global parameter server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--clients", type=int, default=EXPECTED_CLIENTS,
                        help="Number of clients to wait for before FedAvg")
    args = parser.parse_args()
    fl_server.expected_clients = args.clients
    uvicorn.run(app, host=args.host, port=args.port)
