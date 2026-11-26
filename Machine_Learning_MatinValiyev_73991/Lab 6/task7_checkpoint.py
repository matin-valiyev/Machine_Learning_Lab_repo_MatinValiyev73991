import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

def run_task7():
    print("--- Task 7: Checkpoint Recovery ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # --- PART 1: Train for 3 epochs and INTERRUPT ---
    print("\nPart 1: Starting fresh training...")
    model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Train only up to Epoch 3
    for epoch in range(1, 4): 
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Loss: {total_loss/len(loader):.4f}")
        
    # Simulate a crash/interruption by saving a checkpoint
    print("\n[!] Simulating power outage... Saving checkpoint!")
    checkpoint = {
        'epoch': 3,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint, "interrupted_checkpoint.pth")
    
    # --- PART 2: RESUME from checkpoint ---
    print("\nPart 2: Power is back! Loading checkpoint into a BRAND NEW model...")
    
    # We create a completely blank model and optimizer to prove we are loading data
    new_model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
    
    # Load the saved checkpoint file
    loaded_checkpoint = torch.load("interrupted_checkpoint.pth", weights_only=True)
    
    # Inject the saved brain and momentum back into our blank model and optimizer
    new_model.load_state_dict(loaded_checkpoint['model_state'])
    new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
    start_epoch = loaded_checkpoint['epoch'] + 1
    
    print(f"Resuming training from Epoch {start_epoch}...")
    
    # Train for Epochs 4 and 5
    for epoch in range(start_epoch, 6): 
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            new_optimizer.zero_grad()
            loss = criterion(new_model(images), labels)
            loss.backward()
            new_optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Loss: {total_loss/len(loader):.4f}")
        
    print("\nTask 7 complete! Notice how the loss smoothly continued going down.")

if __name__ == '__main__':
    run_task7()