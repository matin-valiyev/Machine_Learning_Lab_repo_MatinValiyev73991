import yaml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity

# --- 1. Setup ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x):
        return self.network(self.flatten(x))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])

# --- 2. The Profiler (Fast Mode) ---
print("Starting the profiler for 10 batches (fast mode)...")

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    data_iter = iter(train_loader)
    
    # We only run 10 loops so the computer doesn't freeze!
    for i in range(10): 
        
        with record_function("1_data_loading"):
            images, labels = next(data_iter)
        
        with record_function("2_forward_pass"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        with record_function("3_backward_pass"):
            optimizer.zero_grad()
            loss.backward()
        
        with record_function("4_optimizer_step"):
            optimizer.step()

# --- 3. Print the Results ---
print("\n--- PROFILING RESULTS ---")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))