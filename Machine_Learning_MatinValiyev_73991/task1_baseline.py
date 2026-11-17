import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import os

# --- 1. The Oversized Baseline Model ---
class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # We design this network specifically to hold about 1 million learnable parameters
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 150), 
            nn.ReLU(),
            nn.Linear(150, 10) 
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def run_task1():
    print("--- Task 1: Baseline CNN ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BigCNN().to(device)

    # 1. Record Parameters
    # This counts every single weight and bias in the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # Prepare Fashion-MNIST Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training for 3 epochs... (This will take a minute or two)")
    for epoch in range(3):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")

    # 2. Record Accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    # 3. Record Latency
    # We pass a fake image through the network 100 times and find the average speed
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    latency_ms = ((time.time() - start_time) / 100) * 1000

    # 4. Record File Size
    # Save the "brain" to the hard drive so we can weigh the file
    torch.save(model.state_dict(), "baseline_cnn.pth")
    size_mb = os.path.getsize("baseline_cnn.pth") / (1024 * 1024)

    print("\n--- Final Baseline Stats ---")
    print(f"Accuracy:          {accuracy:.2f}%")
    print(f"Size on Disk:      {size_mb:.2f} MB")
    print(f"Latency per image: {latency_ms:.2f} milliseconds")

# Windows protection for processing
if __name__ == '__main__':
    run_task1()