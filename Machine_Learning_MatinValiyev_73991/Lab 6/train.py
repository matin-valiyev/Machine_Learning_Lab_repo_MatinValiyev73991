import yaml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import mlflow
import random
import numpy as np
import os

# --- 1. Load the YAML Config ---
# We read the recipe card so the code knows what settings to use
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- 2. Set the Seed for Reproducibility ---
# This forces the computer's random number generator to start at the exact same spot every time
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(config["seed"])

# --- 3. The Lab 3 Model (Simple Neural Network) ---
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.network(self.flatten(x))

# --- 4. Prepare Data and Model ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])

# --- 5. MLflow Experiment Tracking ---
# We open our automatic journal to start recording
mlflow.set_experiment("Lab6_Training_Pipeline")

with mlflow.start_run():
    # Write down our settings (parameters)
    mlflow.log_params(config)
    
    print("Starting training...")
    for epoch in range(config["epochs"]):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Loss: {avg_loss:.4f}")
        
        # Write down our results (metrics)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
    # --- 6. Checkpoint Saving ---
    # Save the brain of the AI to a file
    checkpoint_path = "model_checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    
    # Put the saved file into our MLflow journal (artifacts)
    mlflow.log_artifact(checkpoint_path)
    print("Training complete and logged to MLflow!")