# src/main.py
import torch
from dataset import get_cifar10_datasets
from client import FederatedClient
from server import FederatedServer

def main():
    print("="*50)
    print("  STARTING FEDERATED LEARNING SIMULATION")
    print("="*50)

    # 1. Configuration
    NUM_CLIENTS = 2
    COMMUNICATION_ROUNDS = 3  # How many times they sync with the server
    LOCAL_EPOCHS = 1          # How many epochs each client trains before syncing

    # 2. Prepare the Data Silos
    print("\n[Stage 1] Partitioning Data...")
    client_datasets, testset = get_cifar10_datasets(num_clients=NUM_CLIENTS)
    for i in range(NUM_CLIENTS):
        client_datasets[i] = torch.utils.data.Subset(client_datasets[i], range(100))

    # 3. Initialize the Actors
    print("\n[Stage 2] Initializing Server and Clients...")
    server = FederatedServer()
    
    clients = []
    for i in range(NUM_CLIENTS):
        # Give each client its specific chunk of the dataset
        client = FederatedClient(client_id=i+1, dataset=client_datasets[i])
        clients.append(client)

    # 4. The Federated Training Loop
    print("\n[Stage 3] Beginning Federated Training Loop...")
    
    for round_num in range(1, COMMUNICATION_ROUNDS + 1):
        print(f"\n{'='*20} COMMUNICATION ROUND {round_num} {'='*20}")
        
        # A. Server broadcasts the current global master weights to all clients
        global_weights = server.get_global_weights()
        for client in clients:
            client.set_weights(global_weights)

        # B. Clients train on their private data
        client_updated_weights = []
        for client in clients:
            client.train_local_model(epochs=LOCAL_EPOCHS)
            # Collect the new weights after training
            client_updated_weights.append(client.get_weights())

        # C. Server collects the weights and averages them (FedAvg)
        server.aggregate_weights(client_updated_weights)

    print("="*50)
    print("  FEDERATED SIMULATION COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()