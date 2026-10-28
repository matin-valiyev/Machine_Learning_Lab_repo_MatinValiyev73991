# Lab 1 Report: System Profiling and Model Deployment Viability

## 1. Objective

This experiment evaluates the computational overhead and operational characteristics of deploying machine learning classifiers on constrained computing hardware. Using the Breast Cancer Wisconsin diagnostic dataset, we benchmark two standard algorithms—Logistic Regression and an ensemble Random Forest—to quantify the trade-offs between predictive capability, runtime latency, system memory footprints, and non-volatile storage requirements across Cloud, Edge, Mobile, and TinyML tiers.

## 2. Experimental Setup & Methodology

* Environment Isolation: Configured an isolated Python 3.11 virtual environment (.venv) with pinned dependency specifications logged to results/versions.txt via print_versions.py.
* Data Partitioning: Processed the Breast Cancer Wisconsin dataset into an 70/30 train-test partition using a fixed pseudo-random state (random_state=42).
* Model Training: Fit a baseline linear classifier (Logistic Regression) and an ensemble model (Random Forest, n=100 estimators). Test-set classification performance was exported to results/baseline_accuracy.csv.
* System Profiling Pipeline (measure.py):
* Training Latency: Executed 1 warm-up cycle followed by 5 timed repetitions; computed the median runtime.
* Inference Latency: Evaluated single-instance execution times across 100 iterations; computed the median runtime.
* Memory Consumption: Monitored process Resident Set Size (RSS) during inference execution utilizing psutil.
* Disk Footprint: Serialized binary artifacts via joblib.dump and recorded exact sizes in bytes and kibibytes.



## 3. Empirical Results

### Predictive Performance

* Logistic Regression Test Accuracy: 0.9415
* Random Forest Classifier Test Accuracy: 0.9357

### Computational & Hardware Metrics

* Median Training Latency:
* Logistic Regression: 839.11 ms
* Random Forest: 358.27 ms


* Median Inference Latency (Single Sample):
* Logistic Regression: 0.15 ms
* Random Forest: 7.52 ms


* Peak Inference Memory Footprint:
* Logistic Regression: 103.86 MB
* Random Forest: 104.64 MB


* Serialized Artifact Size:
* Logistic Regression: 1,055 bytes (1.03 KB)
* Random Forest: 290,889 bytes (284.07 KB)



### Deployment Compatibility Analysis

* Cloud Server (Specs: RAM >= 1 GB, Latency <= 100 ms, Storage <= 500 MB):
* Logistic Regression: Compatible
* Random Forest: Compatible


* Edge Gateway (Specs: RAM 256-1024 MB, Latency <= 50 ms, Storage <= 50 MB):
* Logistic Regression: Compatible
* Random Forest: Compatible


* Mobile Device (Specs: RAM 64-256 MB, Latency <= 20 ms, Storage <= 10 MB):
* Logistic Regression: Compatible
* Random Forest: Compatible


* TinyML / Microcontroller (Specs: RAM <= 256 KB, Latency <= 10 ms, Storage <= 100 KB):
* Logistic Regression: Compatible
* Random Forest: Incompatible (Exceeds ROM/flash budget)



## 4. Key Findings & Discussion

1. Model Complexity vs. Generalization: Logistic Regression slightly outperformed the ensemble model (0.9415 vs. 0.9357 test accuracy). For lower-dimensional, structured biomedical tabular data, linear decision boundaries generalize effectively without succumbing to the subtle sample-level variance that can impact complex ensembles.
2. Inference Latency Multipliers: Evaluating a vector dot product and sigmoid activation in Logistic Regression took 0.15 ms per instance, compared to 7.52 ms for traversing 100 trees in the Random Forest—an approximately 50-fold latency reduction critical for hard real-time diagnostic loops.
3. Hardware Storage Envelope: At 1,055 bytes (1.03 KB), Logistic Regression easily satisfies micro-controller flash memory ceilings (< 100 KB). Conversely, the Random Forest model serialized to 290,889 bytes (284.07 KB), immediately disqualifying it from deployment on deeply constrained TinyML hardware without post-training compression or tree-pruning techniques.