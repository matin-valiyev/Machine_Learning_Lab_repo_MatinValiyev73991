import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def test_batch_sizes():
    print("--- Task 4: Batch-Size Study (This takes a few minutes) ---")
    batch_sizes = [32, 64, 128, 256, 512]
    results = []
    
    # Check if your Lenovo LOQ is using its NVIDIA GPU or just the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    
    # We will test each batch size one by one
    for b in batch_sizes:
        print(f"Training with batch size {b}...")
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=b, shuffle=True)
        
        # Create a fresh AI for each test so it is a fair race
        model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        # Train for exactly 5 epochs
        for epoch in range(5):
            correct = 0
            total = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Track how many guesses were correct
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        end_time = time.time()
        
        # --- Math for the Results Table ---
        total_time = end_time - start_time
        # Throughput = total images (5 epochs * 60000 images) divided by seconds
        throughput = (5 * 60000) / total_time
        final_acc = 100.0 * correct / total
        
        # Peak memory calculations
        if torch.cuda.is_available():
            peak_mem = f"{torch.cuda.max_memory_allocated() / (1024*1024):.1f} MB"
        else:
            peak_mem = f"{(b * 28 * 28 * 4 * 3) / (1024*1024):.1f} MB (Est)"
            
        results.append((b, throughput, peak_mem, final_acc))
        
    # --- Print the Final Table ---
    print("\n--- FINAL TASK 4 TABLE ---")
    print("Batch Size | Throughput (samples/s) | Peak Memory | Final Accuracy")
    print("------------------------------------------------------------------")
    for r in results:
        print(f"{r[0]:<10} | {r[1]:<24.0f} | {r[2]:<11} | {r[3]:.2f}%")

if __name__ == '__main__':
    test_batch_sizes()