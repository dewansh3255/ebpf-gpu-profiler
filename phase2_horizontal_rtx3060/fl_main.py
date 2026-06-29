#!/usr/bin/env python3
"""
fl_main.py
Single-process federated-learning simulation (no network).

Useful as a correctness dry-run before deploying the real two-node networked
setup (fl_server.py + fl_client.py). Each simulated client trains on its FULL
CIFAR-10 shard for LOCAL_EPOCHS per round, then the server runs FedAvg.

Usage:
    python3 fl_main.py --arch resnet18 --clients 2 --rounds 3 --local-epochs 3
"""

import argparse

from fl_client import FederatedClient
from fl_dataset import get_cifar10_datasets
from fl_server import FederatedServer


def main():
    parser = argparse.ArgumentParser(description="FL local simulation")
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=3,
                        help="Local epochs per round (>=3 to saturate the workload)")
    args = parser.parse_args()

    print("=" * 50)
    print("  FEDERATED LEARNING SIMULATION")
    print(f"  arch={args.arch} clients={args.clients} "
          f"rounds={args.rounds} local_epochs={args.local_epochs}")
    print("=" * 50)

    # 1. Partition the data into full, non-overlapping shards.
    print("\n[Stage 1] Partitioning data...")
    client_datasets, _ = get_cifar10_datasets(num_clients=args.clients)

    # 2. Initialise server and clients (full shards, no debug slice).
    print("\n[Stage 2] Initialising server and clients...")
    server = FederatedServer(arch=args.arch, expected_clients=args.clients)
    clients = [
        FederatedClient(client_id=i + 1, dataset=client_datasets[i], arch=args.arch)
        for i in range(args.clients)
    ]

    # 3. Federated training loop.
    print("\n[Stage 3] Federated training loop...")
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'=' * 20} COMMUNICATION ROUND {round_num} {'=' * 20}")
        global_weights = server.get_global_weights()
        for client in clients:
            client.set_weights(global_weights)

        client_updated_weights = []
        for client in clients:
            client.train_local_model(epochs=args.local_epochs)
            client_updated_weights.append(client.get_weights())

        server.aggregate_weights(client_updated_weights)

    print("=" * 50)
    print("  FEDERATED SIMULATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
