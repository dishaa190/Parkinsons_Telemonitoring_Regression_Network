**Neural Network for Regression on Parkinson’s Telemonitoring Dataset**

This repository contains an end-to-end implementation of a **fully connected neural network for regression** using **PyTorch**, developed as part of **AI61002: Deep Learning Foundations and Applications (Spring 2026)**.

The project focuses on predicting **motor_UPDRS** and **total_UPDRS** scores from biomedical voice measurements using a carefully designed and evaluated neural network.

**Methodology**

### 1. Dataset Loading & Splitting

* Dataset fetched directly from the **UCI Machine Learning Repository**
* Data split into:

  * **70% Training**
  * **15% Validation**
  * **15% Test**

### 2. Feature Scaling & DataLoaders

* **StandardScaler** applied to input features
* PyTorch `TensorDataset` and `DataLoader` used for efficient batching

### 3. Model Architecture Design

* Fully connected feedforward neural network
* **ReLU activation** in all hidden layers
* **Linear activation** in output layer (regression task)
* Multiple architectures evaluated using validation loss

✅ **Best Architecture Selected**:

Input → [256 → 128 → 64] → Output (2 units)

## Model Analysis

* **Loss Function**: Mean Squared Error (MSE)
* **Optimizer**: Stochastic Gradient Descent (SGD)

  * Momentum = 0.9
* **Learning Rate Scheduler**:

  * ReduceLROnPlateau (reduces LR when validation loss plateaus)

### Model Complexity

* Total trainable parameters reported
* FLOPs computed using:

  * `fvcore`
  * `ptflops`
* Network visualization generated using `torchview`

## Training & Evaluation

* Trained for **100 epochs**
* Training and validation losses tracked and plotted
* Best model saved based on **minimum validation loss**

### Performance Metrics

* **Scatter plots**:
  * Predictions vs Ground Truth (Training Set)
* **R² Score on Test Set**:

  * motor_UPDRS
  * total_UPDRS


## Key Results

* Architecture selection driven by **validation loss**
* Smooth convergence with learning rate scheduling
* Strong regression performance with meaningful R² scores
* Computational cost (FLOPs) analyzed for model efficiency
