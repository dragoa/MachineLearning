##### Definition

To reduce the risk of overfitting, turn to regularization hyperparameters. These help the model generalize better to unseen data.

- **[[Weight decay]]**: a form of L2 regularization. ==Adds a penalty to the loss function based on the magnitude of the model's weights==, targeting the sum of the squares of the weights, which discourages large weights in favor of a simpler model. In PyTorch, this is set via the `weight_decay` parameter in the optimizer.

- **Dropout**: ==randomly disables a portion of neurons during training by setting their activation to zero.== Prevents [[co-adaptation]], encouraging the network to learn more robust features. Common dropout rates range from 0.1 to 0.5, and this value is itself tunable as a hyperparameter.
	
	![[dropout.png|600]]

```python
nn.Dropout(p=0.5) # 50% dropout rate
```

- **Early stopping**: ==halts training when the model's performance on the validation set stops improving.== Prevents overfitting using a `patience` parameter that controls how many epochs to wait after the last improvement before stopping.

	![[early_stopping.png|600]]

- **Batch normalization**: ==normalizes the activations in each layer, usually to a mean of 0 and a standard deviation of 1.== Enhances training speed and stability, and can also act as a form of regularization.

	![[batch_normalization.png|600]]

```python
nn.BatchNorm2d(64) # For a convolutional layer with 64 channels
```
