import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy

# --- 1. The Baseline Model Architecture ---
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

# Helper to test accuracy
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

# Helper to heal the AI (fine-tune)
def finetune(model, loader, device, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

def run_task2():
    print("--- Task 2: Magnitude Pruning ---")
    print("[!] WARNING: This script will take about 15-20 minutes to run.")
    print("[!] It has to re-train the AI 13 separate times to heal it after pruning.\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the healthy brain we saved in Task 1
    base_model = BigCNN().to(device)
    base_model.load_state_dict(torch.load("baseline_cnn.pth", weights_only=True))
    base_acc = evaluate(base_model, test_loader, device)
    print(f"Starting Baseline Accuracy: {base_acc:.2f}%\n")

    sparsity_targets = [0.25, 0.5, 0.75, 0.9, 0.95]
    global_accs = [base_acc]
    layer_accs = [base_acc]
    
    parameters_to_prune = (
        (base_model.features[0], 'weight'),
        (base_model.features[3], 'weight'),
        (base_model.classifier[1], 'weight'),
        (base_model.classifier[3], 'weight'),
    )

    # --- 1. Global Pruning (Searching the whole brain for the weakest numbers) ---
    print("--- Starting GLOBAL Unstructured Pruning ---")
    for s in sparsity_targets:
        model = copy.deepcopy(base_model)
        params = ((model.features[0], 'weight'), (model.features[3], 'weight'),
                  (model.classifier[1], 'weight'), (model.classifier[3], 'weight'))
        
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=s)
        
        print(f"Fine-tuning Global {s*100}% sparsity...")
        finetune(model, train_loader, device, epochs=3)
        acc = evaluate(model, test_loader, device)
        global_accs.append(acc)
        print(f"   Accuracy recovered to: {acc:.2f}%")

    # --- 2. Layer-wise Pruning (Forcing EACH layer to delete its weakest numbers) ---
    print("\n--- Starting LAYER-WISE Unstructured Pruning ---")
    for s in sparsity_targets:
        model = copy.deepcopy(base_model)
        params = ((model.features[0], 'weight'), (model.features[3], 'weight'),
                  (model.classifier[1], 'weight'), (model.classifier[3], 'weight'))
        
        for layer, name in params:
            prune.l1_unstructured(layer, name=name, amount=s)
            
        print(f"Fine-tuning Layer-wise {s*100}% sparsity...")
        finetune(model, train_loader, device, epochs=3)
        acc = evaluate(model, test_loader, device)
        layer_accs.append(acc)
        print(f"   Accuracy recovered to: {acc:.2f}%")

    # --- 3. Structured Pruning (Deleting whole computer chips / filters) ---
    print("\n--- Starting STRUCTURED Pruning ---")
    structured_targets = [0.1, 0.25, 0.5]
    struct_accs = [base_acc] 
    
    for s in structured_targets:
        model = copy.deepcopy(base_model)
        # We only remove entire filters from the Convolutional vision layers
        prune.ln_structured(model.features[0], name="weight", amount=s, n=2, dim=0)
        prune.ln_structured(model.features[3], name="weight", amount=s, n=2, dim=0)
        
        print(f"Fine-tuning Structured {s*100}% sparsity...")
        finetune(model, train_loader, device, epochs=3)
        acc = evaluate(model, test_loader, device)
        struct_accs.append(acc)
        print(f"   Accuracy recovered to: {acc:.2f}%")

    # --- 4. Plot the Sparsity Curves (Graded Evidence) ---
    print("\nSaving sparsity plot to 'sparsity_curves.png'...")
    plt.figure(figsize=(9, 6))
    
    x_unstructured = [0.0] + sparsity_targets
    x_structured = [0.0] + structured_targets
    
    plt.plot(x_unstructured, global_accs, marker='o', label='Global Unstructured', color='blue')
    plt.plot(x_unstructured, layer_accs, marker='s', label='Layer-wise Unstructured', color='green')
    plt.plot(x_structured, struct_accs, marker='^', label='Structured (Conv Filters)', color='red')
    
    plt.title("Model Accuracy vs. Pruning Sparsity")
    plt.xlabel("Sparsity (Fraction of weights removed)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("sparsity_curves.png")
    print("Task 2 complete! Your graded plot has been saved.")

if __name__ == '__main__':
    run_task2()