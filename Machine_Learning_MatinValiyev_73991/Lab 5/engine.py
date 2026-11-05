class Value:
    def __init__(self, data, _children=()):
        # 1. Stores the actual number (data)
        self.data = data            
        
        # 2. Stores the gradient (starts at 0.0)
        self.grad = 0.0             
        
        # 3. Stores the local derivative rule (starts empty)
        self._backward = lambda: None 
        
        # 4. Stores parent nodes (who created this number)
        self._prev = set(_children) 

    # This makes the output readable in the terminal
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    # Task 2: Addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    # Task 2: Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    # Task 2: Powers
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    # Task 2: ReLU
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    # Task 3: Backward pass
    def backward(self):
        # 1. Topologically sort the graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        # 2. Set output gradient to 1
        self.grad = 1.0 

        # 3. Visit nodes in reverse order and calculate blame
        for node in reversed(topo):
            node._backward()

        # Task 4: Sum
    def sum(self):
        # The sum of a single number is just the number itself
        out = Value(self.data, (self,))

        def _backward():
            self.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    # --- TASK 5 MATH EXTENSIONS ---
    
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def sigmoid(self):
        import math
        # Cap the number so the math doesn't crash on huge values
        x = max(-50.0, min(50.0, self.data))
        val = 1.0 / (1.0 + math.exp(-x))
        out = Value(val, (self,))

        # The calculus rule for sigmoid backwards
        def _backward():
            self.grad += (val * (1.0 - val)) * out.grad
        out._backward = _backward

        return out

    def log(self):
        import math
        # Prevent log(0) which breaks the math
        val = max(self.data, 1e-15)
        out = Value(math.log(val), (self,))

        # The calculus rule for log backwards
        def _backward():
            self.grad += (1.0 / val) * out.grad
        out._backward = _backward

        return out
import torch

import time
import torch
import torch.nn as nn

# --- TASK 6: EAGER VS TRACED & CONTIGUITY ---
print("\n--- Task 6: Eager vs Traced Execution ---")

# 1. We create a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.linear(x))

model = SimpleModel()
dummy_input = torch.randn(1024, 512)

# Warm-up (running it a few times to wake up the computer)
for _ in range(10):
    _ = model(dummy_input)

# Time Eager Execution (Normal PyTorch)
start = time.time()
for _ in range(100):
    _ = model(dummy_input)
eager_time = time.time() - start
print(f"Eager Time:  {eager_time:.4f} seconds")

# Trace the model (PyTorch memorizes the exact math steps)
traced_model = torch.jit.trace(model, dummy_input)

# Warm-up traced model
for _ in range(10):
    _ = traced_model(dummy_input)

# Time Traced Execution
start = time.time()
for _ in range(100):
    _ = traced_model(dummy_input)
traced_time = time.time() - start
print(f"Traced Time: {traced_time:.4f} seconds")

# --- 2. Tensor Shape, Dtype, and Contiguity ---
print("\n--- Tensor Contiguity Check ---")

# Create a normal, contiguous tensor
tensor_a = torch.randn(10000, 10000)
print(f"Shape: {tensor_a.shape}, Dtype: {tensor_a.dtype}")
print(f"Is tensor_a contiguous? {tensor_a.is_contiguous()}")

# Create a non-contiguous tensor by flipping it (transposing)
tensor_b = tensor_a.t()
print(f"Is tensor_b (flipped) contiguous? {tensor_b.is_contiguous()}")

# Measure speed of contiguous tensor
start = time.time()
_ = tensor_a.sum()
contig_time = time.time() - start

# Measure speed of non-contiguous tensor
start = time.time()
_ = tensor_b.sum()
non_contig_time = time.time() - start

print(f"Contiguous Sum Time:     {contig_time:.4f} seconds")
print(f"Non-Contiguous Sum Time: {non_contig_time:.4f} seconds")