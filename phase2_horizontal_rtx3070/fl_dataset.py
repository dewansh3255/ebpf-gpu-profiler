# src/dataset.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import warnings

# Hide the NumPy 2.4 VisibleDeprecationWarning from terminal logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.datasets")

def get_cifar10_datasets(num_clients=2):
    """
    Downloads CIFAR-10 and splits it into non-overlapping 
    datasets for a given number of clients.
    """
    # Standard ResNet/CIFAR-10 transforms with augmentation for higher accuracy
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("Fetching CIFAR-10 Dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    dataset_size = len(trainset)
    silo_size = dataset_size // num_clients
    lengths = [silo_size] * num_clients
    lengths[-1] += dataset_size - sum(lengths)

    client_datasets = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))
    
    print(f"Split {dataset_size} images into {num_clients} client silos.")
    return client_datasets, testset

if __name__ == "__main__":
    client_data, test_data = get_cifar10_datasets(num_clients=2)
    print(f"Client 1 dataset size: {len(client_data[0])}")
    print(f"Client 2 dataset size: {len(client_data[1])}")