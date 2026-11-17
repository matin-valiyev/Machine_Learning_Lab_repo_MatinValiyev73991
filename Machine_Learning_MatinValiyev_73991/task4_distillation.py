import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import os

# --- 1. The Models ---
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

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_task4():
    print("--- Task 4: Knowledge Distillation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 1. Load the Big Teacher
    teacher = BigCNN().to(device)
    teacher.load_state_dict(torch.load("baseline_cnn.pth", weights_only=True))
    teacher.eval()
    
    # --- 2. Train a Plain Student (No Teacher) ---
    print("\nTraining Plain Student (Without Teacher) for 3 epochs...")
    plain_student = SmallCNN().to(device)
    
    params_count = sum(p.numel() for p in plain_student.parameters())
    print(f"Student Parameters: {params_count:,} (Nice and small!)")
    
    optimizer = torch.optim.Adam(plain_student.parameters(), lr=0.001)
    
    for epoch in range(3):
        plain_student.train()
        for images, labels in train_loader:
            # FIXED: Properly unpack and move both to the device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(plain_student(images), labels)
            loss.backward()
            optimizer.step()
    plain_acc = evaluate(plain_student, test_loader, device)
    print(f"Plain Student Accuracy: {plain_acc:.2f}%")
    
    # --- 3. Train a Distilled Student (WITH Teacher) ---
    print("\nTraining Distilled Student (WITH Teacher) for 3 epochs...")
    distilled_student = SmallCNN().to(device)
    optimizer = torch.optim.Adam(distilled_student.parameters(), lr=0.001)
    
    T = 4.0
    alpha = 0.5
    
    for epoch in range(3):
        distilled_student.train()
        for images, labels in train_loader:
            # FIXED: Properly unpack and move both to the device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            student_logits = distilled_student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
            kd_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1)
            )
            
            loss = alpha * ce_loss + (1 - alpha) * (T * T) * kd_loss
            loss.backward()
            optimizer.step()
            
    distilled_acc = evaluate(distilled_student, test_loader, device)
    print(f"Distilled Student Accuracy: {distilled_acc:.2f}%")
    
    # --- 4. Record Student Specs ---
    torch.save(distilled_student.state_dict(), "student_cnn.pth")
    size_mb = os.path.getsize("student_cnn.pth") / (1024 * 1024)
    
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    start = time.time()
    for _ in range(10): distilled_student(dummy_input) 
    start = time.time()
    with torch.no_grad():
        for _ in range(100): distilled_student(dummy_input)
    lat_ms = ((time.time() - start) / 100) * 1000
    
    print(f"\nStudent Size on Disk:      {size_mb:.2f} MB")
    print(f"Student Latency per image: {lat_ms:.2f} milliseconds")

if __name__ == '__main__':
    run_task4()