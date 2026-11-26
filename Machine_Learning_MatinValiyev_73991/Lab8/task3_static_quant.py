import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import os

# --- 1. Quantization-Ready Architecture ---
# Same exact layers as BigCNN, plus QuantStub (entry) and DeQuantStub (exit)
class BigCNNQuant(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 7 * 7, 150), nn.ReLU(), nn.Linear(150, 10)
        )
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)               # Convert input image from FP32 -> INT8
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)             # Convert final output from INT8 -> FP32
        return x

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

def run_task3():
    print("--- Task 3: Post-Training Static Quantization ---")

    # 1. Pick the supported quantization backend
    supported = torch.backends.quantized.supported_engines
    backend = next(b for b in ["fbgemm", "x86", "onednn", "qnnpack"] if b in supported)
    torch.backends.quantized.engine = backend
    print(f"Selected Quantization Backend: {backend}")

    # Load Fashion-MNIST Train (for calibration) and Test (for evaluation)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

    # Pick 500 random training images for calibration
    torch.manual_seed(42)
    indices = torch.randperm(len(train_dataset))[:500]
    calib_subset = torch.utils.data.Subset(train_dataset, indices)
    calib_loader = torch.utils.data.DataLoader(calib_subset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Load the FP32 weights into our Quantization-Ready model
    model = BigCNNQuant()
    model.load_state_dict(torch.load("baseline_cnn.pth", weights_only=True, map_location="cpu"))
    model.eval()

    # 3. Attach observers (thermometers) to measure activation ranges
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(model, inplace=True)

    # 4. Calibrate using the 500 random training images
    print("Calibrating on 500 random training images...")
    with torch.no_grad():
        for images, _ in calib_loader:
            _ = model(images)

    # 5. Convert the whole model to INT8
    torch.ao.quantization.convert(model, inplace=True)
    torch.save(model.state_dict(), "static_quant_cnn.pth")

    # 6. Evaluate Accuracy, Size, and Latency
    print("Evaluating Static Quantized model...")
    sq_acc = evaluate(model, test_loader)
    sq_size = os.path.getsize("static_quant_cnn.pth") / (1024 * 1024)
    sq_lat = measure_latency(model)

    print("\n--- Task 3 Results ---")
    print("Model           | Accuracy | Size (MB) | Latency (ms)")
    print("------------------------------------------------------")
    print(f"Static Quant    | {sq_acc:.2f}%   | {sq_size:.2f} MB   | {sq_lat:.2f} ms")

if __name__ == '__main__':
    run_task3()