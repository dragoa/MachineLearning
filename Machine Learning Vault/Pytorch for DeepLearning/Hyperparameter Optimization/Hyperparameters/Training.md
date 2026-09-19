##### Definition

**Optimizer** An optimizer is an algorithm that tweaks the model's weights and biases during backpropagation to minimize the loss function and improve performance.

```
Optimizer
├─ Gradient Descent
│   ├─ Stochastic Gradient Descent (SGD)
│   ├─ Mini-Batch Gradient Descent
│   └─ SGD with Momentum
└─ Adaptive Gradient Descent
    ├─ AdaDelta
    ├─ RMSProp
    └─ Adam
```

- **Gradient Descent** — adjusts weights and biases based on a gradient to minimize the loss function.
- **RMSProp** (Root Mean Squared Propagation) — adjusts the learning rate for each parameter individually, using a moving average of recent gradients.
- **Adam** (Adaptive Moment Estimation) — one of the most widely used optimizers. Computes adaptive learning rates for each parameter using momentum. ==Straightforward to implement, computationally efficient, and requires little memory.==

**Batch size** The number of samples processed before updating the model's internal parameters. Hardware limitations like RAM and GPU capacity can dictate your choice here.

- **Smaller batch size** → may take longer per epoch, but uses less memory and introduces noise to gradient estimates, which can help escape local minima.
- **Larger batch size** → smoother and quicker epoch updates, but requires more memory and can get stuck in local minima, potentially affecting convergence.

==Starting with a batch size of 32 or 64==, depending on your data, is often a good approach.

**Learning rate & schedulers** See [[Learning Rate Schedulers]].
[[Adam]]
[[Sgd]]
