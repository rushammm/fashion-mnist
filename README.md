# Fashion-MNIST — from-scratch MLP → PyTorch CNN

Two implementations of a Fashion-MNIST classifier, built to learn deep learning from first principles before relying on frameworks.

1. **MLP in pure NumPy** — every line of forward prop, backprop, and the Adam optimizer written by hand. No PyTorch, no autograd.
2. **CNN in PyTorch** — same dataset, same goal, but using `nn.Module`, autograd, and `torch.optim.Adam`.

The point isn't the dataset (Fashion-MNIST is a teaching benchmark). The point is going through the mechanics twice — once by hand, once with the framework — so the framework calls are never a black box.

## Results

| Model | Framework | Test accuracy | Parameters |
|---|---|---|---|
| MLP (1 hidden layer, 128 units) | pure NumPy | **87.62%** | ~101k |
| CNN (2 conv blocks → FC) | PyTorch | **88.68%** | ~20k |

The CNN beats the MLP on accuracy with **~5× fewer parameters** — a concrete demonstration of why CNNs are more sample-efficient on images (weight sharing + spatial inductive bias).

## MLP architecture (NumPy)

```
Input 784 (28×28 flattened) → Hidden 128 (sigmoid) → Output 10 (softmax)
```

- Loss: categorical cross-entropy
- Optimizer: **Adam from scratch** — first/second moments, bias correction, per-weight learning rates, all implemented manually
- Mini-batch size 128, shuffled per epoch, 20 epochs
- Backward pass written by hand using chain rule on `dZ2 = A2 - y` (the elegant softmax + CE gradient)

## CNN architecture (PyTorch)

```
input (1, 28, 28)
  → conv1 (1→16, 3×3, pad=1) → ReLU → MaxPool 2×2  → (16, 14, 14)
  → conv2 (16→32, 3×3, pad=1) → ReLU → MaxPool 2×2 → (32, 7, 7)
  → flatten → FC (1568 → 10) → CrossEntropyLoss (log_softmax + NLL fused)
```

- Loss: `nn.CrossEntropyLoss` (model returns raw logits; softmax is fused into the loss for numerical stability)
- Optimizer: `torch.optim.Adam`, lr 1e-3, 5 epochs
- Padding=1 preserves spatial size at each conv (so spatial only changes at pool layers)

## What this project demonstrates

- **Forward + backward propagation by hand** — not just calling `loss.backward()`. The NumPy notebook makes every gradient explicit.
- **Adam optimizer from scratch** — running moments, bias correction, the `w -= lr * m_hat / (sqrt(v_hat) + eps)` update rule, all implemented and explained.
- **The math behind PyTorch idioms** — why `nn.CrossEntropyLoss` takes raw logits (numerical stability), why `optimizer.zero_grad()` is needed (gradients accumulate by default), what `model.eval()` and `torch.no_grad()` actually do.
- **Why CNNs > MLPs on images** — concretely shown via the parameter/accuracy comparison. The CNN's weight sharing and local connectivity beat the MLP with a fraction of the parameters.
- **Working with shape arithmetic** — every layer's output shape is traced and verified, including the conv output-size formula `(H + 2P - K)/S + 1`.

## Run

```bash
pip install numpy matplotlib datasets torch torchvision
jupyter notebook fashion_mnist_mlp.ipynb   # NumPy MLP
jupyter notebook fashion_mnist_cnn.ipynb   # PyTorch CNN
```

## Files

- `fashion_mnist_mlp.ipynb` — pure NumPy MLP with hand-derived Adam
- `fashion_mnist_cnn.ipynb` — PyTorch CNN with detailed comments mapping each line back to its NumPy equivalent
