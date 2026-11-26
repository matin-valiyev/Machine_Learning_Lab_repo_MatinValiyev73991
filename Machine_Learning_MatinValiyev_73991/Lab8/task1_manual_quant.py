import torch

def quantize_per_tensor(w, num_bits):
    # Max integer value for symmetric quantization: 127 (8-bit), 7 (4-bit), 1 (2-bit)
    qmax = (2 ** (num_bits - 1)) - 1
    
    # 1. Calculate ONE scale for the entire tensor (zero-point = 0)
    scale = torch.max(torch.abs(w)) / qmax
    if scale == 0:
        scale = 1e-8  # Avoid dividing by zero
        
    # 2. Map to integer (Quantize)
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    
    # 3. Map back to decimal (Dequantize)
    w_reconstructed = q * scale
    
    # 4. Measure Mean Squared Error (MSE)
    mse = torch.mean((w - w_reconstructed) ** 2).item()
    return mse

def quantize_per_channel(w, num_bits):
    qmax = (2 ** (num_bits - 1)) - 1
    
    # 1. Calculate a SEPARATE scale for each output channel (dim=0)
    # Reshape w to [num_channels, everything_else] to find max per channel
    w_flat = w.view(w.shape[0], -1)
    max_vals = torch.max(torch.abs(w_flat), dim=1)[0]
    scales = max_vals / qmax
    scales[scales == 0] = 1e-8
    
    # Reshape scales so it aligns with w's dimensions: [64, 1, 1, 1]
    scales = scales.view(-1, 1, 1, 1)
    
    # 2. Map to integer (Quantize)
    q = torch.clamp(torch.round(w / scales), -qmax, qmax)
    
    # 3. Map back to decimal (Dequantize)
    w_reconstructed = q * scales
    
    # 4. Measure Mean Squared Error (MSE)
    mse = torch.mean((w - w_reconstructed) ** 2).item()
    return mse

def run_task1():
    print("--- Task 1: Manual Tensor Quantization ---")
    
    # Load the saved weights from our Lab 7 baseline model
    state_dict = torch.load("baseline_cnn.pth", weights_only=True, map_location="cpu")
    
    # Grab the weight tensor of the first Conv2d layer (shape: [64, 1, 3, 3])
    w = state_dict["features.0.weight"]
    print(f"Loaded weight tensor 'features.0.weight' with shape: {tuple(w.shape)}\n")
    
    print("Bit-Width | Per-Tensor MSE  | Per-Channel MSE | Improvement")
    print("-------------------------------------------------------------")
    
    for bits in [8, 4, 2]:
        mse_tensor = quantize_per_tensor(w, bits)
        mse_channel = quantize_per_channel(w, bits)
        improvement = ((mse_tensor - mse_channel) / mse_tensor) * 100
        print(f"{bits}-bit     | {mse_tensor:.8f}      | {mse_channel:.8f}      | {improvement:.1f}% lower error")

if __name__ == '__main__':
    run_task1()