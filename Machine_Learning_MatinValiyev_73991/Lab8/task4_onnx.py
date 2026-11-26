import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
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

def run_task4():
    print("--- Task 4: ONNX Export & ONNX Runtime ---")

    # 1. Load the trained FP32 baseline model
    model = BigCNN()
    model.load_state_dict(torch.load("baseline_cnn.pth", weights_only=True, map_location="cpu"))
    model.eval()

    # Create a dummy input image (batch=1, channels=1, height=28, width=28)
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 1, 28, 28)

    # 2. Export to ONNX
    onnx_path = "baseline_cnn.onnx"
    print(f"Exporting model to '{onnx_path}'...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False
    )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Export successful! ONNX file size: {onnx_size:.2f} MB\n")

    # 3. Numerical Validation (Check outputs match within 1e-4)
    with torch.no_grad():
        pytorch_out = model(dummy_input).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": dummy_input.numpy()}
    ort_out = session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pytorch_out - ort_out))
    matches = np.allclose(pytorch_out, ort_out, atol=1e-4)

    print("--- Numerical Validation ---")
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Matches within 1e-4?     {matches}\n")

    # 4. Measure Latency: PyTorch vs ONNX Runtime
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
    pt_lat = ((time.time() - start) / 100) * 1000

    for _ in range(10):
        _ = session.run(None, ort_inputs)
    start = time.time()
    for _ in range(100):
        _ = session.run(None, ort_inputs)
    ort_lat = ((time.time() - start) / 100) * 1000

    print("--- Latency Comparison ---")
    print(f"PyTorch CPU Latency:      {pt_lat:.2f} ms")
    print(f"ONNX Runtime CPU Latency: {ort_lat:.2f} ms")
    print(f"Speedup:                  {pt_lat / ort_lat:.2f}x")

if __name__ == '__main__':
    run_task4()