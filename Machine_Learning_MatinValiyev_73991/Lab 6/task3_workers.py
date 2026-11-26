import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def test_workers():
    print("Starting Task 3: Testing num_workers... (This will take a minute)")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # download=False because we already have the data
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)

    workers = [0, 1, 2, 4, 8]
    times = []

    for w in workers:
        # We create a DataLoader with a specific number of background helpers (num_workers)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=w)
        
        start_time = time.time()
        # We just loop through the images as fast as possible to see how long it takes
        for images, labels in loader:
            pass 
        end_time = time.time()
        
        duration = end_time - start_time
        times.append(duration)
        print(f"Helpers (num_workers={w}): {duration:.4f} seconds")

    # --- Plotting the Results ---
    plt.figure(figsize=(8, 5))
    plt.plot(workers, times, marker='o', linestyle='-', color='red', linewidth=2)
    plt.title("Data Loading Time vs. Number of Workers")
    plt.xlabel("Number of Workers (num_workers)")
    plt.ylabel("Time to load 1 epoch (seconds)")
    plt.grid(True)
    
    # Save the plot as an image file
    plt.savefig("workers_plot.png")
    print("\nPlot saved successfully as 'workers_plot.png'!")

# CRITICAL FOR WINDOWS: 
# Windows will crash if you use multiple workers without this exact line protecting the code!
if __name__ == '__main__':
    test_workers()