import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Forces the first dedicated GPU
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import uvicorn
import io
import zlib  # --- NEW: Added for Network Compression ---
import torchvision.datasets as datasets
import torchvision.transforms as transforms

app = FastAPI(title="Federated Learning Global Server")

class FederatedServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.global_model = resnet18(num_classes=10)
        self.global_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.global_model.maxpool = nn.Identity()
        self.global_model.to(self.device)
        
        self.expected_clients = 2
        self.received_weights = []
        self.current_round = 1

        # --- Setup Global Test Dataset ---
        print("[Server] Loading Global Test Set for Evaluation...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(testset, batch_size=100, shuffle=False)

    def evaluate_global_model(self):
        """Tests the master model against the test set."""
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"\n✅ [EVALUATION] Round {self.current_round-1} Global Accuracy: {accuracy:.2f}%")
        print("-" * 50)

    def aggregate_weights(self):
        print(f"\n[Server] Aggregating Round {self.current_round}...")
        global_dict = self.global_model.state_dict()
        
        for key in global_dict.keys():
            temp_tensor = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for client_dict in self.received_weights:
                temp_tensor += client_dict[key].to(self.device)
            global_dict[key] = temp_tensor / len(self.received_weights)
            
        self.global_model.load_state_dict(global_dict)
        self.current_round += 1
        self.received_weights = []
        
        # Run evaluation after aggregation
        self.evaluate_global_model()

fl_server = FederatedServer()

@app.get("/status")
async def get_status():
    # Tells the clients exactly what round the server is currently on
    return {"current_round": fl_server.current_round}

@app.get("/get_weights")
async def get_weights():
    buffer = io.BytesIO()
    torch.save(fl_server.global_model.state_dict(), buffer)
    
    # --- NEW: Compress the Master Model before broadcasting ---
    compressed_data = zlib.compress(buffer.getvalue(), level=3)
    return Response(content=compressed_data, media_type="application/octet-stream")

@app.post("/upload_weights")
async def upload_weights(file: UploadFile = File(...)):
    contents = await file.read()
    
    # --- NEW: Decompress the incoming client weights ---
    decompressed_data = zlib.decompress(contents)
    buffer = io.BytesIO(decompressed_data)
    
    client_state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    fl_server.received_weights.append(client_state_dict)
    
    print(f"[Network] Received weights from client ({len(fl_server.received_weights)}/{fl_server.expected_clients})")
    
    if len(fl_server.received_weights) == fl_server.expected_clients:
        fl_server.aggregate_weights()
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)