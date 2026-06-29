#!/usr/bin/env python3
"""
21_ml_workload.py
Group 21 - GRS Project Part A

PyTorch DDP (DistributedDataParallel) training workload for profiling.
Trains ResNet-18 on CIFAR-10 dataset across multiple GPUs.
This script is the actual ML workload whose system-level behavior
is profiled by our eBPF/eGPU tools.

Usage:
    # Single GPU
    python3 21_ml_workload.py --gpus 1 --epochs 5 --batch-size 128

    # Multi-GPU (DDP) — 2 GPUs
    torchrun --nproc_per_node=2 21_ml_workload.py --gpus 2 --epochs 5

Authors: Dewansh Khandelwal, Palak Mishra, Sanskar Goyal, Yash Nimkar, Kunal Verma
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms


def setup_distributed():
    """Initialize the distributed process group for DDP."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_cifar10_dataloaders(batch_size, num_workers=4, distributed=False):
    """Create CIFAR-10 train and test data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transform_test
    )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(trainset)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return trainloader, testloader, train_sampler


def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch):
    """Train for one epoch and return metrics."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    epoch_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_time = time.time() - epoch_start
    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(trainloader)

    return {
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "train_accuracy": accuracy,
        "epoch_time_sec": epoch_time,
        "samples_per_sec": total / epoch_time,
    }


def evaluate(model, testloader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    eval_start = time.time()

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    eval_time = time.time() - eval_start

    return {
        "test_loss": test_loss / len(testloader),
        "test_accuracy": 100.0 * correct / total,
        "eval_time_sec": eval_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ML Workload: ResNet-18 on CIFAR-10 - Group 21"
    )
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use (default: 1)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size per GPU (default: 128)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--output", type=str,
                        default="21_training_results.json",
                        help="Output JSON file for training metrics")
    args = parser.parse_args()

    # Setup
    distributed = args.gpus > 1
    rank, world_size, local_rank = setup_distributed() if distributed else (0, 1, 0)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU (will be slow).")
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{local_rank}")

    is_main = (rank == 0)

    if is_main:
        print("=" * 60)
        print("ML WORKLOAD: ResNet-18 on CIFAR-10")
        print("=" * 60)
        print(f"GPUs: {args.gpus}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total effective batch size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"GPU Memory: "
                  f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        print("=" * 60)

    # Data
    trainloader, testloader, train_sampler = get_cifar10_dataloaders(
        batch_size=args.batch_size, distributed=distributed
    )

    # Model
    model = torchvision.models.resnet18(num_classes=10)
    # Modify first conv layer for CIFAR-10 (32x32 images instead of 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training loop
    results = {
        "config": {
            "gpus": args.gpus,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "model": "ResNet-18",
            "dataset": "CIFAR-10",
        },
        "epochs": [],
    }

    total_start = time.time()

    for epoch in range(args.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        epoch_metrics = train_one_epoch(
            model, trainloader, criterion, optimizer, device, epoch
        )

        # Evaluate
        eval_metrics = evaluate(model, testloader, criterion, device)
        epoch_metrics.update(eval_metrics)

        scheduler.step()

        results["epochs"].append(epoch_metrics)

        if is_main:
            print(f"Epoch {epoch + 1}/{args.epochs}: "
                  f"Train Loss={epoch_metrics['train_loss']:.4f}, "
                  f"Train Acc={epoch_metrics['train_accuracy']:.1f}%, "
                  f"Test Acc={eval_metrics['test_accuracy']:.1f}%, "
                  f"Time={epoch_metrics['epoch_time_sec']:.1f}s, "
                  f"Throughput={epoch_metrics['samples_per_sec']:.0f} samples/s")

    total_time = time.time() - total_start

    results["total_time_sec"] = total_time

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Final test accuracy: "
              f"{results['epochs'][-1]['test_accuracy']:.1f}%")
        print(f"{'=' * 60}")

        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
