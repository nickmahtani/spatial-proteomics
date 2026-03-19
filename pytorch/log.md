# PyTorch Learning Log

## 19 March 2026

### Deep Learning: PyTorch MLPs, DataLoaders, and Training Mechanics

Continued through Raschka's PyTorch tutorial — moved from running a forward pass to understanding the full data pipeline.

**What I did:**

- Ran a forward pass through a 3-layer MLP (50→30→20→1, 731 parameters) with random input and traced the `grad_fn` chain back through the computational graph
- Built a custom `Dataset` with `__getitem__` and `__len__`, fed it through a `DataLoader` with batch_size=2 — 5 samples split into 3 batches (2, 2, 1)
- Explored softmax with `dim=1` to convert logits to class-membership probabilities

**Key observations:**

- Models return raw logits, not probabilities — `F.cross_entropy` handles softmax internally for numerical stability. You only apply softmax yourself at inference
- `drop_last=True` avoids noisy gradients from a small final batch, but with only 1 epoch those dropped samples are gone forever
- `num_workers>0` can actually slow things down — process startup overhead costs more than the loading itself
- Parameter count scales fast: `Linear(50, 30)` alone has 1530 params (50×30 weights + 30 biases). Each neuron connects to every input from the previous layer

## 13 March 2026

### Deep Learning: PyTorch Autograd and Logistic Regression from Scratch

Started Sebastian Raschka's PyTorch in One Hour tutorial — building up from tensors to a working logistic regression classifier.

**What I did:**

- Built a logistic regression classifier from raw tensors: z = x1*w1 + b → sigmoid → binary cross entropy loss
- Computed gradients manually via chain rule (dloss/dw1 = -0.0915, dloss/db = -0.0832) then verified with `loss.backward()`
- Debugged `torch.Tensor` vs `torch.tensor` — capital T doesn't accept `requires_grad`
- Hit the "backward through graph a second time" error and learned the graph gets freed after `.backward()`
- Explored leaf vs intermediate nodes — `.grad` is only stored on leaf tensors, intermediate nodes need `.retain_grad()`

**Key observations:**

- The sigmoid output `a` is the model's prediction, not the raw score `z`. The loss compares `a` (probability) against `y` (label) — they don't need to be the same units
- Bias is just a learnable offset that lets the decision boundary sit anywhere, not just at the origin
- Binary cross entropy = -log(a) when y=1. Confident and wrong gets punished exponentially harder than being unsure
- ReLU's gradient is 0 or 1 — no vanishing gradient problem. One ReLU is just a kink, but many ReLUs with different weights across layers collectively approximate complex curves
- Python OOP basics clicked: `self` connects methods to their instance, `super().__init__()` sets up the parent Module's internals so parameter tracking works
