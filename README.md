# FashionMNIST-Neural-Network-with-PyTorch

This repository contains a PyTorch implementation of a fully-connected neural network trained on the FashionMNIST dataset. The project demonstrates deep learning workflows including model training, evaluation, GPU acceleration, optimizer experimentation, and performance profiling.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [GPU vs CPU](#gpu-vs-cpu)
- [Optimizer Experiments](#optimizer-experiments)
- [Fine-Tuning](#fine-tuning)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Project Overview
This project builds a multi-layer perceptron (MLP) to classify images from the FashionMNIST dataset into 10 categories. It covers:
- Data preprocessing and augmentation
- Model building using PyTorch
- Training and evaluation loops
- GPU acceleration with CUDA
- Performance profiling (latency, throughput, memory)
- Experimentation with different optimizers and hyperparameters

---

## Dataset
**FashionMNIST** is a dataset of 60,000 training images and 10,000 test images of 28x28 grayscale fashion products in 10 categories.  
- Categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot  
- Dataset is automatically downloaded via `torchvision.datasets.FashionMNIST`  
- Images are normalized to [0,1] range using `ToTensor()`

---

## Model Architecture
- Fully-connected feedforward neural network (MLP)  
- Input layer: 28x28 flattened to 784 nodes  
- Hidden layers:  
  - Linear(784 → 512) + ReLU  
  - Linear(512 → 512) + ReLU  
- Output layer: Linear(512 → 10)  
- Loss function: CrossEntropyLoss  
- Optimizers: SGD, Adam (experimented)

---

## Training & Evaluation
- Training loop with backpropagation and optimizer step  
- Evaluation loop with `torch.no_grad()` for test set  
- Metrics tracked:  
  - Accuracy  
  - Average loss  
- Epochs: 15  
- Batch size: 64  

---

## GPU vs CPU
- Model can run on CPU or GPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`)  
- Performance metrics observed:
  - Average latency per batch: ~0.32 ms (GPU)  
  - Throughput: ~197,596 images/sec (GPU)  
  - Peak GPU memory usage: 0.03 GB  

---

## Optimizer Experiments
- SGD with momentum  
- Adam optimizer  
- Learning rate scheduling to improve convergence  
- Observed differences in training speed and final accuracy

---

## Fine-Tuning
- Model architecture adjustments: added non-linear layers  
- Hyperparameter tuning: learning rate, batch size, optimizer selection  
- Data augmentation: random horizontal flips, resizing  

---

## Usage
Clone the repository:

```bash
git clone https://github.com/<your-username>/fashionmnist-pytorch.git
cd fashionmnist-pytorch
