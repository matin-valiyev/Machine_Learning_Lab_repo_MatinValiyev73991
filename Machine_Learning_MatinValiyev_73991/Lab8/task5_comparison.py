import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os

# --- 1. Model Architectures ---
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
        return self.dequant(self.classifier(self.features(self.quant(x))))

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# --- 2. Helper Functions ---
def load_pruned_weights(path):
    """Bakes PyTorch pruning masks (weight_orig * weight_mask) into clean weights."""
    raw = torch.load(path, weights_only=True, map_location="cpu")
    clean = {}
    for key in list(raw.keys()):
        if key.endswith("_orig"):
            base = key[:-5]
            clean[base] = raw[key] * raw[base + "_mask"]
        elif key.endswith("_mask"):
            continue
        else:
            clean[key] = raw[key]
    model = BigCNN()
    model.load_state_dict(clean)
    model.eval()
    return model

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
    dummy = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        start = time.time()
        for _ in range(100):
            _ = model(dummy)
    return ((time.time() - start) / 100) * 1000

def get_size_mb(model, filename):
    torch.save(model.state_dict(), filename)
    return os.path.getsize(filename) / (1024 * 1024)

def run_task5():
    print("--- Task 5: Full Model-Variant Comparison ---")

    # Auto-select supported quantization backend
    supported = torch.backends.quantized.supported_engines
    backend = next(b for b in ["fbgemm", "x86", "onednn", "qnnpack"] if b in supported)
    torch.backends.quantized.engine = backend
    print(f"Using Quantization Backend: {backend}\n")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 1. Baseline
    baseline = BigCNN()
    baseline.load_state_dict(torch.load("baseline_cnn.pth", weights_only=True, map_location="cpu"))
    baseline.eval()

    # 2. Pruned (90%)
    pruned = load_pruned_weights("pruned_cnn.pth")

    # 3. Distilled Student
    distilled = SmallCNN()
    distilled.load_state_dict(torch.load("student_cnn.pth", weights_only=True, map_location="cpu"))
    distilled.eval()

    # 4. Dynamic Quant
    dyn_quant = torch.ao.quantization.quantize_dynamic(baseline, {nn.Linear}, dtype=torch.qint8)

    # 5. Static Quant (calibrated on 500 random training images)
    torch.manual_seed(42)
    calib_indices = torch.randperm(len(train_dataset))[:500]
    calib_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, calib_indices), batch_size=32)
    stat_quant = BigCNNQuant()
    stat_quant.load_state_dict(torch.load("baseline_cnn.pth", weights_only=True, map_location="cpu"))
    stat_quant.eval()
    stat_quant.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(stat_quant, inplace=True)
    with torch.no_grad():
        for images, _ in calib_loader:
            _ = stat_quant(images)
    torch.ao.quantization.convert(stat_quant, inplace=True)

    # 6. Pruned + Dynamic Quant
    pruned_quant = torch.ao.quantization.quantize_dynamic(pruned, {nn.Linear}, dtype=torch.qint8)

    # 7. Distilled + Dynamic Quant
    distilled_quant = torch.ao.quantization.quantize_dynamic(distilled, {nn.Linear}, dtype=torch.qint8)

    variants = [
        ("Baseline",        baseline,        "baseline_cnn.pth"),
        ("Pruned (90%)",    pruned,          "pruned_clean.pth"),
        ("Distilled",       distilled,       "student_cnn.pth"),
        ("Dynamic Quant",   dyn_quant,       "dynamic_quant_cnn.pth"),
        ("Static Quant",    stat_quant,      "static_quant_cnn.pth"),
        ("Pruned+Quant",    pruned_quant,    "pruned_quant_cnn.pth"),
        ("Distilled+Quant", distilled_quant, "distilled_quant_cnn.pth"),
    ]

    results = []
    print("Benchmarking all 7 variants... (Takes about 30 seconds)\n")
    for name, model, fname in variants:
        acc = evaluate(model, test_loader)
        size = get_size_mb(model, fname)
        lat = measure_latency(model)
        results.append((name, acc, size, lat))

    # Print Master Table
    print("Variant          | Accuracy | Size (MB) | Latency (ms)")
    print("-------------------------------------------------------")
    for name, acc, size, lat in results:
        print(f"{name:<16} | {acc:>6.2f}%  | {size:>6.2f} MB | {lat:>6.2f} ms")

    # --- 3. Plot Accuracy vs Latency and Accuracy vs Size ---
    names = [r[0] for r in results]
    accs  = [r[1] for r in results]
    sizes = [r[2] for r in results]
    lats  = [r[3] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Accuracy vs Latency
    ax1.scatter(lats, accs, color='blue', s=80)
    for i, txt in enumerate(names):
        ax1.annotate(txt, (lats[i], accs[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax1.set_title("Accuracy vs. Latency")
    ax1.set_xlabel("Latency (ms per image)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True)

    # Subplot 2: Accuracy vs Size
    ax2.scatter(sizes, accs, color='green', s=80)
    for i, txt in enumerate(names):
        ax2.annotate(txt, (sizes[i], accs[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax2.set_title("Accuracy vs. Model Size")
    ax2.set_xlabel("Model Size (MB)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("task5_tradeoffs.png")
    print("\nPlot saved successfully as 'task5_tradeoffs.png'!")

if __name__ == '__main__':
    run_task5()