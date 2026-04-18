"""
Self-Pruning Neural Network for CIFAR-10 Image Classification
==============================================================
Implements learnable gate parameters that dynamically prune weights during training.
Each weight has an associated gate value (0-1); the network learns which connections
are unnecessary by pushing gate values toward zero via L1 sparsity regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────────
# Device Configuration
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. Custom Prunable Linear Layer
# ─────────────────────────────────────────────
class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        """Kaiming uniform for weights; gate_scores start slightly positive (gates ≈ 0.73)."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)          

        pruned_weight = self.weight * gates               

        # Standard linear transformation with pruned weights
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from graph for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ─────────────────────────────────────────────
# 2. Self-Pruning Network Architecture
# ─────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    """
    Feedforward network with PrunableLinear layers for CIFAR-10 (32×32×3 → 10 classes).

    Architecture:
        Flatten → PrunableLinear(3072, 512) → ReLU → PrunableLinear(512, 10)
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)   
        self.relu = nn.ReLU()
        self.fc2 = PrunableLinear(512, 10)              

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def prunable_layers(self):
        """Yield all PrunableLinear sub-modules for easy iteration."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


# ─────────────────────────────────────────────
# 3. Sparsity-Regularised Loss Function
# ─────────────────────────────────────────────
def compute_total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model: SelfPruningNet,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_loss = F.cross_entropy(logits, targets)

    sparsity_loss = torch.tensor(0.0, device=logits.device)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        sparsity_loss = sparsity_loss + gates.abs().sum()

    total_loss = cls_loss + lam * sparsity_loss
    return total_loss, cls_loss, sparsity_loss


# ─────────────────────────────────────────────
# 4. Data Loading
# ─────────────────────────────────────────────
def get_dataloaders(batch_size: int = 64):
    """Download CIFAR-10 and return train/test DataLoaders with normalisation."""
    # CIFAR-10 channel means and standard deviations
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, test_loader


# ─────────────────────────────────────────────
# 5. Training Loop
# ─────────────────────────────────────────────
def train_model(
    model: SelfPruningNet,
    train_loader: DataLoader,
    lam: float,
    num_epochs: int = 10,
    lr: float = 1e-3,
) -> list[dict]:
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_total = epoch_cls = epoch_spar = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)

            total_loss, cls_loss, spar_loss = compute_total_loss(
                logits, labels, model, lam
            )
            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_cls   += cls_loss.item()
            epoch_spar  += spar_loss.item()
            n_batches   += 1

        metrics = {
            "epoch":         epoch,
            "total_loss":    epoch_total / n_batches,
            "cls_loss":      epoch_cls   / n_batches,
            "sparsity_loss": epoch_spar  / n_batches,
        }
        history.append(metrics)

        print(
            f"  Epoch {epoch:>2}/{num_epochs} | "
            f"Total: {metrics['total_loss']:.4f} | "
            f"CE: {metrics['cls_loss']:.4f} | "
            f"Sparsity: {metrics['sparsity_loss']:.2f}"
        )

    return history


# ─────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────
def evaluate_model(
    model: SelfPruningNet,
    test_loader: DataLoader,
    sparsity_threshold: float = 1e-2,
) -> dict:
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    accuracy = 100.0 * correct / total

    # Gate-based sparsity analysis
    all_gates = []
    with torch.no_grad():
        for layer in model.prunable_layers():
            all_gates.append(layer.get_gates().cpu().numpy().ravel())

    all_gates = np.concatenate(all_gates)
    sparsity  = 100.0 * (all_gates < sparsity_threshold).sum() / len(all_gates)

    return {"accuracy": accuracy, "sparsity": sparsity, "all_gates": all_gates}


# ─────────────────────────────────────────────
# 7. Experiment Runner
# ─────────────────────────────────────────────
def run_experiment(
    lam: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 10,
) -> dict:
    """Train and evaluate a fresh model for a given λ value."""
    print(f"\n{'='*60}")
    print(f"  Running experiment: λ = {lam}")
    print(f"{'='*60}")

    model = SelfPruningNet().to(DEVICE)

    t0 = time.time()
    train_model(model, train_loader, lam=lam, num_epochs=num_epochs)
    elapsed = time.time() - t0

    results = evaluate_model(model, test_loader)
    results["lam"]     = lam
    results["elapsed"] = elapsed

    print(
        f"\n  → Test Accuracy:  {results['accuracy']:.2f}%  |  "
        f"Sparsity: {results['sparsity']:.2f}%  |  "
        f"Time: {elapsed:.1f}s"
    )
    return results


# ─────────────────────────────────────────────
# 8. Visualisation
# ─────────────────────────────────────────────
def plot_gate_distributions(all_results: list[dict], save_path: str = "gate_distributions.png"):
    """
    Plot a histogram of gate value distributions for each λ experiment side-by-side.
    """
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colours = ["#2563EB", "#16A34A", "#DC2626"]  # blue, green, red

    for ax, result, colour in zip(axes, all_results, colours):
        gates = result["all_gates"]
        ax.hist(gates, bins=50, color=colour, alpha=0.80, edgecolor="white", linewidth=0.4)
        ax.axvline(x=0.01, color="black", linestyle="--", linewidth=1.2, label="Threshold (0.01)")
        ax.set_title(f"λ = {result['lam']}\nAcc: {result['accuracy']:.1f}%  |  "
                     f"Sparse: {result['sparsity']:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Gate Value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Distribution of Gate Values Across PrunableLinear Layers",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGate distribution plot saved to: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 9. Summary Table
# ─────────────────────────────────────────────
def print_summary_table(all_results: list[dict]):
    """Print a formatted summary table of all experiments."""
    print("\n" + "=" * 50)
    print(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level (%)':>20}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['lam']:<12} {r['accuracy']:>14.2f}% {r['sparsity']:>19.2f}%")
    print("=" * 50)


# ─────────────────────────────────────────────
# 10. Main Entry Point
# ─────────────────────────────────────────────
def main():
    # Hyperparameters
    BATCH_SIZE  = 64
    NUM_EPOCHS  = 10        # Increase for better accuracy; 10 is fast for demonstration
    LAMBDAS     = [0.0001, 0.001, 0.01]
    THRESHOLD   = 1e-2      # Gate value below which a weight is considered pruned

    print("Loading CIFAR-10 dataset …")
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # Run one experiment per lambda
    all_results = []
    for lam in LAMBDAS:
        result = run_experiment(lam, train_loader, test_loader, num_epochs=NUM_EPOCHS)
        all_results.append(result)

    # Print summary table
    print_summary_table(all_results)

    # Visualise gate distributions
    plot_gate_distributions(all_results, save_path="gate_distributions.png")


if __name__ == "__main__":
    main()
