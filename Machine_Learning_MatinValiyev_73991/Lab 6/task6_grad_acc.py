import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def run_task6():
    print("--- Task 6: Gradient Accumulation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    
    # We use a small batch size of 64
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Emulate batch size 256 (64 images * 4 steps = 256 images)
    accumulation_steps = 4 
    
    print("Training with Batch 64 and Accumulation 4 (emulating Batch 256)...")
    
    # Train for 2 epochs to see the behavior
    for epoch in range(2):
        total_loss = 0
        
        # 1. Start with a clean slate
        optimizer.zero_grad() 
        
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # 2. Divide the loss by 4 because we are adding 4 of them together
            loss = criterion(outputs, labels) / accumulation_steps 
            
            # 3. Calculate blame and let it pile up (accumulate) without updating yet!
            loss.backward() 
            
            # 4. Only update the weights (learn) every 4th step
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad() # Reset the blame for the next 4 batches
                
            total_loss += loss.item() * accumulation_steps
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(loader):.4f}")
        
    print("\nGradient Accumulation successfully emulated Batch 256!")

if __name__ == '__main__':
    run_task6()