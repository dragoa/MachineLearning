### What is a Scheduler?
  
==A scheduler **dynamically adjusts the learning rate during training**==, typically starting high to learn quickly, then lowering to fine-tune. PyTorch provides three common ones:
  
```
1. StepLR → Fixed interval decay
2. ReduceLROnPlateau → Reactive decay (responds to performance)
3. CosineAnnealingLR → Smooth cosine curve decay
```

### Scheduler 1: StepLR

**Idea:** ==Multiply the learning rate by `gamma` every `step_size` epochs, regardless of how the model is performing.==

![StepLR](images/steplr.png)

**Parameters:**

- `step_size`: how many epochs between each reduction

- `gamma`: multiplication factor at each step (e.g., 0.2 = reduce to 20%)

**Example:** LR=1.0, gamma=0.2, step_size=3

```
Epoch 0-2: LR = 1.0
Epoch 3-5: LR = 1.0 × 0.2 = 0.2
Epoch 6-8: LR = 0.2 × 0.2 = 0.04
...
```

**PyTorch:**

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
# At the end of each epoch:
scheduler.step()
```

✅ Simple and predictable
⚠️ Reduces blindly — doesn't respond to actual model performance
### Scheduler 2: ReduceLROnPlateau

**Idea:** ==Only reduce the learning rate when the model stops improving.== Watches a metric and reacts.

![ReduceLROnPlateau](images/reduce_lr_plateau.png)

**Parameters:**
- `mode` — `'max'` if tracking accuracy (higher = better), `'min'` if tracking loss (lower = better)
- `factor` — multiplication factor when reducing (e.g., 0.2)
- `patience` — how many epochs to wait without improvement before reducing

**Example:** patience=3, factor=0.2, mode='max'

```
Epochs 1-3: Accuracy improves → no change to LR
Epoch 4-6: Accuracy stalls for 3 epochs (patience) → LR × 0.2
Epoch 7-9: Accuracy improves again → no change
Epoch 10-12: Accuracy stalls again → LR × 0.2
```

**PyTorch:**

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
	optimizer, mode='max', factor=0.2, patience=3
)

# Pass the metric you're monitoring at end of each epoch:
scheduler.step(val_acc)
```

✅ Adaptive — only reacts when actually needed
✅ Won't reduce if the model is still learning
⚠️ Requires monitoring a validation metric each epoch

### Scheduler 3: CosineAnnealingLR

**Idea:** ==Smoothly reduce the learning rate following a cosine curve== from its initial value down to a minimum, no sudden drops.

![CosineAnnealingLR](images/cosine_annealing.png)

**Parameters:**
- `T_max` — total number of epochs (length of the cosine cycle)
- `eta_min` — minimum learning rate at the end of training

**PyTorch:**

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
optimizer, T_max=n_epochs, eta_min=0.0002)

# At the end of each epoch:
scheduler.step()
```

✅ Smoothest transition — no abrupt drops
✅ Generally produces stable, gradual convergence
⚠️ Less responsive to actual performance than ReduceLROnPlateau

The scheduler's own settings (`step_size`, `patience`, `gamma`, `eta_min`) are themselves hyperparameters that need tuning. Adding a scheduler means more things to optimize. Later in this module, you'll see automated strategies to handle this.