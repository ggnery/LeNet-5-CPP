# 🔥 LeNet-5 C++ (Mini-torch)

![MNIST Dataset Example](./images/MNIST_dataset_example.png)

A lightweight, educational deep learning framework built from scratch in C++ using PyTorch's LibTorch backend. This project implements a complete neural network framework with support for convolutional layers, dense layers, various activation functions, and training on the MNIST dataset.

## What is Mini-Torch?

Mini-Torch is a hands-on implementation of fundamental deep learning concepts, perfect for understanding how neural networks work under the hood. Unlike high-level frameworks that abstract away the details, Mini-Torch lets you see exactly how forward propagation, backpropagation, and gradient descent work at a low level.

The project demonstrates a complete LeNet-5 style Convolutional Neural Network (CNN) architecture trained on the classic MNIST handwritten digit dataset, achieving the core functionality you'd expect from a modern deep learning framework.

## Features

### Core Architecture
- **Sequential Model Builder**: Fluent API for building neural networks layer by layer
- **Automatic Differentiation**: Complete backpropagation implementation from scratch
- **GPU Support**: Optional CUDA acceleration for faster training
- **Multiple Layer Types**:
  - Dense (Fully Connected) layers with Xavier initialization
  - Convolutional layers with learnable kernels and bias
  - Max Pooling for spatial downsampling
  - Reshape layers for tensor manipulation

### Activation Functions
- **Tanh**: Hyperbolic tangent activation
- **Sigmoid**: Classic sigmoid activation  
- **Softmax**: For multi-class classification output

### Loss Functions & Optimization
- **Cross-Entropy Loss**: Perfect for classification tasks
- **Mean Squared Error**: For regression problems
- **Stochastic Gradient Descent**: Simple yet effective optimization

### Data Processing
- **MNIST Dataset Loader**: Direct binary file reading (.idx format)
- **Data Preprocessing**: Automatic normalization and formatting
- **Model Evaluation**: Built-in accuracy calculation

## Architecture Overview

The implemented LeNet-5 style network follows this architecture:

```
Input (1×32×32) 
    ↓
Conv2D (6 kernels, 5×5) → Tanh → MaxPool (2×2) → Tanh
    ↓
Conv2D (16 kernels, 5×5) → Tanh → MaxPool (2×2) → Tanh  
    ↓
Conv2D (120 kernels, 5×5) → Tanh
    ↓
Reshape → Dense (120→84) → Tanh → Dense (84→10) → Softmax
    ↓
Output (10 classes)
```

## Prerequisites

- **CMake** 3.28.3 or higher
- **GCC 12** (specifically gcc-12 and g++-12 for Ubuntu 24.04)
- **C++17** compatible compiler
- **LibTorch** 2.8.0 (automatically downloaded by build script)
- **CUDA** (optional, for GPU acceleration)

### Installing GCC-12 on Ubuntu 24.04

```bash
# Update package list
sudo apt update

# Install GCC-12 and G++-12
sudo apt install gcc-12 g++-12

# Verify installation
gcc-12 --version
g++-12 --version
```

## Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd mini-torch
```

### 2. Download and Setup LibTorch
The project includes a convenient script to download the appropriate LibTorch version:

```bash
chmod +x build_libs.sh
./build_libs.sh
```

You'll be prompted to choose your preferred version:
- **CPU Only** (recommended for learning)
- **CUDA 12.9** (latest GPU support)
- **CUDA 12.8** (stable GPU support)  
- **CUDA 12.6** (legacy GPU support)

### 3. Build the Project
```bash
mkdir build && cd build

# For CPU version
cmake ..
make

# For GPU version (if you have CUDA)
cmake -DWITH_CUDA=ON ..
make
```

### 4. Download MNIST Dataset
The MNIST dataset files should be placed in the `data/` directory:
- `train-images.idx3-ubyte` - Training images
- `train-labels.idx1-ubyte` - Training labels  
- `t10k-images.idx3-ubyte` - Test images
- `t10k-labels.idx1-ubyte` - Test labels

### 5. Run Training
```bash
./build/mini_torch
```

You should see output like:
```
Train images sizes: [60000, 1, 32, 32]
Train labels sizes: [60000, 10]
Test images sizes: [10000, 1, 32, 32] 
Test labels sizes: [10000, 10]
Loss in epoch 1: 2.1543
Loss in epoch 2: 1.8234
Final model accuracy: 0.892
```

## Code Structure

```
mini-torch/
├── src/
│   ├── main.cpp              # Main training loop
│   ├── dataset/              # MNIST data loading
│   │   ├── mnist.hpp/cpp     # Binary file readers
│   └── network/              # Neural network implementation
│       ├── include/          # Header files
│       ├── network.cpp       # High-level training interface
│       ├── sequential.cpp    # Sequential model container
│       ├── dense.cpp         # Fully connected layers
│       ├── convolutional.cpp # CNN layers
│       ├── activation.cpp    # Activation functions
│       ├── max_pooling.cpp   # Pooling operations
│       ├── reshape.cpp       # Tensor reshaping
│       └── softmax.cpp       # Softmax activation
├── data/                     # MNIST dataset files
├── libs/                     # LibTorch installation
├── build_libs.sh            # LibTorch setup script
└── CMakeLists.txt           # Build configuration
```

## Educational Value

This project is perfect for:

- **Learning Deep Learning Fundamentals**: See how gradients flow through networks
- **Understanding C++ in ML**: Modern C++17 features in machine learning context
- **Framework Design**: How to structure a ML library with clean abstractions
- **Performance Optimization**: CPU vs GPU computation trade-offs
- **Computer Vision**: Classic CNN architectures and their implementation

## Understanding the Code

### The Layer Abstraction
Every component inherits from the `Layer` base class:
```cpp
class Layer {
    virtual torch::Tensor forward(torch::Tensor input) = 0;
    virtual torch::Tensor backward(torch::Tensor output_gradient, double eta) = 0;
};
```

### Building Networks
The fluent builder pattern makes network construction intuitive:
```cpp
Sequential network = Sequential::builder()
    .add<Convolutional>(input_shape, kernel_size, num_kernels)
    .add<Tanh>()
    .add<Dense>(input_size, output_size)
    .build();
```

### Training Loop
The training process is straightforward:
1. **Forward Pass**: Compute predictions
2. **Loss Calculation**: Compare with ground truth
3. **Backward Pass**: Compute gradients
4. **Parameter Update**: Apply gradient descent

## Customization

Want to experiment? Try:

- **New Architectures**: Add ReLU, Dropout, or Batch Normalization layers
- **Different Datasets**: Adapt the data loader for CIFAR-10 or custom datasets
- **Optimizers**: Implement Adam, RMSprop, or momentum-based SGD
- **New Loss Functions**: Add focal loss, huber loss, or custom objectives

## Common Issues

**Build Errors**: Ensure you have GCC-12 and C++17 support with the correct LibTorch version
**CUDA Issues**: Verify your CUDA version matches the LibTorch variant
**Memory Errors**: The current implementation loads full dataset into memory
**Performance**: CPU training is slow - consider GPU version for larger experiments

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyTorch Team** for the excellent LibTorch C++ API
- **MNIST Database** creators for the classic dataset
- **LeNet Architecture** by Yann LeCun and colleagues

---

*Omnia cum Iesu, nihil sine Maria ❤️* <br>
*everything for Christ nothing without holy Mary ❤️*
