# src/client.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Forces the first dedicated GPU
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import requests
import io
import time
import zlib  # Added for network payload compression

class FederatedClient:
    def __init__(self, client_id, dataset, batch_size=64, lr=0.01):
        self.client_id = client_id
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        # ACCURACY UPGRADE: Added weight_decay to prevent overfitting
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def train_local_model(self, epochs=3): # ACCURACY UPGRADE: Train for 3 epochs locally per round
        self.model.train()
        print(f"--- Client {self.client_id} starting local training for {epochs} epoch(s) on {self.device} ---")
        
        for epoch in range(epochs):
            running_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}")
            
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_loss = running_loss / len(self.dataloader)
            print(f"Client {self.client_id} | Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")


# --- NETWORK LOGIC ---
def run_network_client(client_id, server_url="http://192.168.52.110:8000"):
    from dataset import get_cifar10_datasets
    print(f"Client {client_id} initializing data...")
    client_datasets, _ = get_cifar10_datasets(num_clients=2)
    
    # 5000 images per client for realistic training
    my_data = torch.utils.data.Subset(client_datasets[client_id - 1], range(100))
    client = FederatedClient(client_id=client_id, dataset=my_data)
    
    rounds = 2
    for r in range(1, rounds + 1):
        print(f"\n=== ROUND {r} ===")
        
        # --- SYNCHRONIZATION BARRIER ---
        print(f"Waiting for Server to reach Round {r}...")
        while True:
            try:
                status_response = requests.get(f"{server_url}/status").json()
                if status_response["current_round"] == r:
                    print(f"Server is ready for Round {r}!")
                    break
                else:
                    time.sleep(2) # Wait for slower nodes
            except Exception as e:
                print("Error connecting to server:", e)
                time.sleep(2)

        # 1. DOWNLOAD & DECOMPRESS global weights from Server
        print("Downloading and decompressing global weights...")
        try:
            response = requests.get(f"{server_url}/get_weights")
            if response.status_code == 200:
                decompressed_data = zlib.decompress(response.content)
                buffer = io.BytesIO(decompressed_data)
                global_weights = torch.load(buffer, map_location="cpu", weights_only=True)
                client.set_weights(global_weights)
            else:
                print("Failed to reach server. Exiting.")
                break
        except Exception as e:
            print(f"Connection error: {e}")
            break

        # 2. TRAIN locally (Increased to 3 Epochs for accuracy)
        client.train_local_model(epochs=1)

        # 3. COMPRESS & UPLOAD new weights back to Server
        print("Compressing and uploading trained weights to server...")
        local_weights = client.get_weights()
        
        buffer = io.BytesIO()
        torch.save(local_weights, buffer)
        buffer.seek(0)
        
        # Compress the 45MB payload down to ~35MB to reduce Wi-Fi ACKs
        compressed_data = zlib.compress(buffer.getvalue(), level=3)
        compressed_buffer = io.BytesIO(compressed_data)
        
        files = {"file": ("weights.pth", compressed_buffer, "application/octet-stream")}
        requests.post(f"{server_url}/upload_weights", files=files)
        
        print(f"Round {r} upload complete. Entering sync barrier for next round...")
        # Old time.sleep(5) completely removed!

if __name__ == "__main__":
    import sys
    c_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_network_client(client_id=c_id)