# The-Self-Pruning-Neural-Network
A PyTorch neural network that uses learnable gate parameters to automatically prune its own weights while it is being trained. Each weight has a gate value between 0 and 1. The model is encouraged to get rid of unnecessary connections by using a combined CrossEntropy and L1 sparsity loss instead of a separate pruning step.

# Self-Pruning Neural Network — CIFAR-10

Implements a neural network that prunes its own weights during training
using learnable gate parameters. Built with PyTorch.

## How to run
pip install torch torchvision matplotlib numpy
python self_pruning_cifar10.py

## Results
Lambda 0.0001 → ~47% accuracy, ~12% sparsity
Lambda 0.001  → ~44% accuracy, ~55% sparsity
Lambda 0.01   → ~38% accuracy, ~90% sparsity
