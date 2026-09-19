Penalizing large weights by adding the squares of the weights to the loss function

$$
Loss = Error(Y - \hat{Y}) + \lambda \sum_{i=1}^{n} w_i^2
$$
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```
