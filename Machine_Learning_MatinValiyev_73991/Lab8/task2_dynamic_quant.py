import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import os

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

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def measure_latency(model):
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
    return ((time.time() - start) / 100) * 1000

def run_task2():
    print("--- Task 2: Post-Training Dynamic Quantization ---")
    
    # 1. Check supported quantization backends and pick the best Intel/AMD one
    supported = torch.backends.quantized.supported_engines
    print(f"Supported Quantization Backends on this PC: {supported}")
    
    backend = None
    for candidate in ["fbgemm", "x86", "onednn", "qnnpack"]:
        if candidate in supported:
            backend = candidate
            break
            
    if backend is None:
        raise RuntimeError(f"No supported quantization backend found! Available: {supported}")
        
    torch.backends.quantized.engine = backend
    print(f"Selected Quantization Backend: {torch.backends.quantized.engine}\n")
    
    # Load Fashion-MNIST Test Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 2. Load the FP32 Baseline Model (CPU only for PyTorch quantization)
    fp32_model = BigCNN()
    fp32_model.load_state_dict(torch.load("baseline_cnn.pth", weights_only=True, map_location="cpu"))
    fp32_model.eval()
    
    # 3. Apply Dynamic Quantization to all Linear layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    # Save the quantized model to measure its size on disk
    torch.save(quantized_model.state_dict(), "dynamic_quant_cnn.pth")
    
    # 4. Measure Baseline vs Dynamic Quantized
    print("Evaluating models... (Takes a few seconds)")
    fp32_acc = evaluate(fp32_model, test_loader)
    fp32_size = os.path.getsize("baseline_cnn.pth") / (1024 * 1024)
    fp32_lat = measure_latency(fp32_model)
    
    dq_acc = evaluate(quantized_model, test_loader)
    dq_size = os.path.getsize("dynamic_quant_cnn.pth") / (1024 * 1024)
    dq_lat = measure_latency(quantized_model)
    
    print("\n--- Task 2 Results ---")
    print("Model             | Accuracy | Size (MB) | Latency (ms)")
    print("--------------------------------------------------------")
    print(f"Baseline (FP32)   | {fp32_acc:.2f}%   | {fp32_size:.2f} MB   | {fp32_lat:.2f} ms")
    print(f"Dynamic Quant     | {dq_acc:.2f}%   | {dq_size:.2f} MB   | {dq_lat:.2f} ms")

if __name__ == '__main__':
    run_task2()