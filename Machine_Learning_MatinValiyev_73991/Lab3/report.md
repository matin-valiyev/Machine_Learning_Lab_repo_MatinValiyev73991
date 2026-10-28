# Lab 3 Report: Deep Learning Topology Profiling and Computational Efficiency

## 1. Goal

The objective of this investigation is to evaluate structural trade-offs, arithmetic intensity, and latency characteristics across three foundational deep learning paradigms—Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and a Vision Transformer encoder (ViT / Attention)—implemented in PyTorch. Using the Fashion-MNIST image classification task, we analyze capacity (trainable parameter volume), arithmetic workload (floating-point operations, FLOPs), execution speed across isolated and batched inference, and convergence within constrained training intervals.

## 2. Method

* **Data Pipeline:** Utilized Fashion-MNIST containing 60,000 training instances and 10,000 evaluation samples over 10 categories. Grayscale intensity was standardized using mean 0.2860 and standard deviation 0.3530. A train-validation partition allocated 55,000 samples to optimization and 5,000 samples to validation monitoring.
* **Model Formulations:**
1. **Fully Connected Network (MLP):** Unrolls the 28x28 2D input into a 784-dimensional vector, projecting through dense stages of 784 to 256, 256 to 128, and a terminal 128 to 10 projection.
2. **Convolutional Architecture (CNN):** Employs two sequential 3x3 convolution stages (1 to 16 feature maps, followed by 16 to 32 feature maps), interspaced with 2x2 max-pooling reductions, terminating in a dense mapping to the 10-class logit space.
3. **Patch-Based Self-Attention (Vision Transformer):** Segments each 28x28 input into a grid of 16 patches (7x7 spatial resolution). Projects each patch to a 64-dimensional embedding, routes token sequences through 2 Transformer encoder blocks (4 self-attention heads per layer), pools sequence representations via global temporal averaging, and outputs 10 class logits.


* **Experimental Protocol:**
* Quantified structural weights and parameters using `torchinfo.summary`.
* Derived theoretical multiply-accumulate operations (FLOPs) analytically for selected layers and cross-referenced overall pipeline workloads using `thop.profile`.
* Recorded CPU execution latency for single-sample inference (batch size 1, 100 iterations) and batched execution (batch size 128, 20 iterations), computing the median runtime.
* Trained all candidate architectures for 5 epochs with the Adam optimizer (learning rate 0.001, mini-batch size 128).



## 3. Results

### Quantitative Summary

**1. Multi-Layer Perceptron (MLP):**

* Trainable Parameters: 235,146 dials
* Total Floating-Point Operations (thop): 469,504
* Single-Sample Inference Latency (Batch 1): 0.204 ms
* Batched Inference Latency (Batch 128): 0.702 ms
* Top-1 Validation Accuracy (5 Epochs): 87.83%

**2. Convolutional Neural Network (CNN):**

* Trainable Parameters: 20,490 dials
* Total Floating-Point Operations (thop): 2,063,488
* Single-Sample Inference Latency (Batch 1): 0.495 ms
* Batched Inference Latency (Batch 128): 18.781 ms
* Top-1 Validation Accuracy (5 Epochs): 89.75%

**3. Vision Transformer (Attention):**

* Trainable Parameters: 70,794 dials
* Total Floating-Point Operations (thop): 1,182,976
* Single-Sample Inference Latency (Batch 1): 2.212 ms
* Batched Inference Latency (Batch 128): 10.859 ms
* Top-1 Validation Accuracy (5 Epochs): 83.86%

### Theoretical Derivations vs. Profiling Output

* **First Dense Layer (784 -> 256):**
$2 \times 784 \times 256 = 401,408\text{ FLOPs}$. Incorporating the downstream transformations (256 -> 128 and 128 -> 10) aggregates to 469,504 FLOPs, aligning identically with the output reported by `thop`.
* **First Convolutional Layer (1 -> 16, 3x3 filter):**
$2 \times 9 \times 1 \times 16 \times 28 \times 28 = 225,792\text{ FLOPs}$. Accumulating the second convolutional stage, pooling, and linear projection yields 2,063,488 FLOPs, reflecting dual-operation multiply-add calculations.
* **Attention Operator (16 token patches, dimension 64):**
The linear Query-Key-Value projection evaluates to $2 \times 16 \times 64 \times 192 = 393,216\text{ FLOPs}$, while cross-token sequence similarity demands $2 \times 16 \times 16 \times 64 = 32,768\text{ FLOPs}$.
* **Root Cause of Numerical Variance:** Discrepancies between handwritten sub-layer formulas and programmatic profiler metrics stem from framework overhead; `thop` measures compound activation functions, residual additions, LayerNorm routines, and MLP feedforward blocks, whereas manual derivations isolate the core linear algebra projections.

## 4. Hardware Behavior and Memory Access Patterns

1. **Spatial inductive bias grants the CNN optimal parameter economy:**
The CNN achieved peak classification accuracy (89.75%) with just 20,490 parameters—an 11.5x reduction relative to the MLP. Weight sharing allows identical 3x3 kernels to slide across spatial coordinates, preserving locality while remaining small enough to reside almost entirely within low-level CPU cache hierarchies during inference.
2. **The MLP suffers from excessive memory movement:**
With 235,146 parameters, the MLP dedicates independent weights to each spatial coordinate without translational invariance. Flattening 2D arrays strips geometric adjacency, forcing the architecture to ingest large parameter buffers from main memory on every pass. While batch-1 arithmetic is basic and registers low latency (0.204 ms), memory bandwidth scaling becomes a bottleneck under heavier workloads.
3. **Quadratic attention overhead penalizes low-data, small-scale regimes:**
The ViT formulation yielded the lowest score (83.86%) and the highest single-item inference latency (2.212 ms). Pairwise token comparison generates non-sequential memory lookups, and without hardcoded spatial priors, Transformers require substantial dataset scale and extended pre-training schedules to discover topological patterns that convolutions assume inherently.

## 5. Conclusions

* **Parameter count does not determine task accuracy:** The MLP maintains the largest parameter footprint (235k) but falls short of the compact CNN (20k) due to the absence of spatial locality.
* **Convolutions offer the best balance of speed and footprint:** Delivering highest accuracy at minimal model capacity, the CNN remains the ideal candidate for edge, mobile, and microcontroller deployment.
* **Self-attention demands scale to amortize structural overhead:** Splitting low-resolution 28x28 inputs into 16 tokens and evaluating cross-attention introduces latency penalties that yield diminishing returns on constrained training budgets.