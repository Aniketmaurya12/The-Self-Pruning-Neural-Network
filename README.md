# The-Self-Pruning-Neural-Network
A PyTorch neural network that uses learnable gate parameters to automatically prune its own weights while it is being trained. Each weight has a gate value between 0 and 1. The model is encouraged to get rid of unnecessary connections by using a combined CrossEntropy and L1 sparsity loss instead of a separate pruning step.
