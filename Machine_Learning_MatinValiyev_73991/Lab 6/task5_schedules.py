import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def train_model(optimizer_name, scheduler_name):
    # Setup standard model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 1. Choose the Optimizer
    if optimizer_name == "SGD+Momentum":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else: # Adam
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
    # 2. Choose the Learning Rate Scheduler
    if scheduler_name == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

    print(f"Training: {optimizer_name} with {scheduler_name}...")
    
    # Train for exactly 10 epochs
    for epoch in range(10):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        # Schedulers need to be updated after every epoch
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(avg_loss)
        else:
            scheduler.step()
            
    # Calculate final accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100.0 * correct / total

def run_task5():
    print("--- Task 5: Optimizers and Schedulers (Takes a few minutes) ---")
    
    # We test four different combinations
    configs = [
        ("SGD+Momentum", "StepLR"),
        ("SGD+Momentum", "CosineAnnealingLR"),
        ("SGD+Momentum", "ReduceLROnPlateau"),
        ("Adam", "StepLR") 
    ]
    
    results = []
    for opt, sch in configs:
        acc = train_model(opt, sch)
        results.append((opt, sch, acc))
        
    print("\n--- FINAL TASK 5 TABLE ---")
    print("Optimizer      | Scheduler            | Final Accuracy")
    print("------------------------------------------------------")
    for r in results:
        print(f"{r[0]:<14} | {r[1]:<20} | {r[2]:.2f}%")

if __name__ == '__main__':
    run_task5()