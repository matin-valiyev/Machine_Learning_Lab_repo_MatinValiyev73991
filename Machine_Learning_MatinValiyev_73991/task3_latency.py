import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
import os
import zipfile

# --- 1. The Baseline Architecture ---
class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 7 * 7, 150), nn.ReLU(), nn.Linear(150, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def run_task3():
    print("--- Task 3: Latency & Storage Comparison ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the heavy baseline model
    dense_model = BigCNN().to(device)
    dense_model.load_state_dict(torch.load("baseline_cnn.pth", weights_only=True))
    
    # Create a pruned version (90% global unstructured sparsity)
    pruned_model = BigCNN().to(device)
    pruned_model.load_state_dict(torch.load("baseline_cnn.pth", weights_only=True))
    params = ((pruned_model.features[0], 'weight'), (pruned_model.features[3], 'weight'),
              (pruned_model.classifier[1], 'weight'), (pruned_model.classifier[3], 'weight'))
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.9)
    
    # --- 1. Measure Latency ---
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    # Warmup the computer
    for _ in range(10): 
        dense_model(dummy_input)
        pruned_model(dummy_input)
        
    start = time.time()
    for _ in range(100): dense_model(dummy_input)
    dense_lat = ((time.time() - start) / 100) * 1000
    
    start = time.time()
    for _ in range(100): pruned_model(dummy_input)
    pruned_lat = ((time.time() - start) / 100) * 1000
    
    print(f"Dense Baseline Latency: {dense_lat:.2f} ms")
    print(f"Pruned Model Latency:   {pruned_lat:.2f} ms")
    
    # --- 2. Measure Storage Size ---
    torch.save(pruned_model.state_dict(), "pruned_cnn.pth")
    
    # To see the true sparse size, we zip both files.
    # Compression algorithms easily shrink millions of zeros!
    with zipfile.ZipFile('dense.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write('baseline_cnn.pth')
    with zipfile.ZipFile('sparse.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write('pruned_cnn.pth')
        
    size_dense = os.path.getsize('baseline_cnn.pth') / (1024 * 1024)
    size_dense_zipped = os.path.getsize('dense.zip') / (1024 * 1024)
    size_sparse_zipped = os.path.getsize('sparse.zip') / (1024 * 1024)
    
    print(f"\nDense Model Size (Raw):     {size_dense:.2f} MB")
    print(f"Dense Model (Compressed):   {size_dense_zipped:.2f} MB")
    print(f"Pruned Model (Compressed):  {size_sparse_zipped:.2f} MB")

if __name__ == '__main__':
    run_task3()