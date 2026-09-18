At its core, optimization is about **finding the best possible value of a function**, a maximum or a minimum.

In machine learning, this usually means adjusting parameters or architecture to improve some objective metric: accuracy, speed, or memory efficiency.  

![[optimization.png|350]]

# 1. Evaluation Metrics

Different metrics tell you different things about your model. The metric you optimize determines what your model gets good at.

### Confusion Matrix

All classification metrics are based on **four fundamental outcomes**:

```
                              REAL
                       ┌─────────────┬─────────────┐
                       │  Positive   │  Negative   │
          ┌────────────┼─────────────┼─────────────┤
          │ Positive   │     TP      │     FP      │
PREDICTED │            │ True        │ False       │
          │            │ Positive    │ Positive    │
          ├────────────┼─────────────┼─────────────┤
          │ Negative   │    FN       │     TN      │
          │            │ False       │ True        │
          │            │ Negative    │ Negative    │
          └────────────┴─────────────┴─────────────┘

```

- TP (True Positive): The model predicted positive, and the real class is positive ✅

- FP (False Positive): The model predicted positive, but the real class is negative ❌
	*Example: The model incorrectly diagnoses a healthy plant as diseased.*
	
- FN (False Negative): The model predicted negative, but the real class is positive ❌
	*Example: The model fails to detect an infected plant.*

- TN (True Negative): The model predicted negative, and the real class is negative ✅

#### Binary vs Multiclass
  
**Binary:** One "positive" class you want to detect vs "negative" (everything else)

- Examples: Spam/Not spam, Disease/Healthy, Fraud/Legitimate

**Multiclass:** Multiple distinct classes (no inherent "positive")

- Examples: Dog/Cat/Bird, Digits 0-9
- Calculate metrics per class (treating each as "positive" vs rest), then combine using macro / weighted / micro averaging

---
### Accuracy
![[Accuracy#Definition]]
### Precision
![[Precision#Definition]]

### Recall
![[Recall#Definition]]

### F1 Score
![[F1 Score#Definition]]


---
# 2. Computing Metrics in PyTorch

PyTorch's `torchmetrics` library computes these metrics efficiently during evaluation.
  
- **Binary:** `task="binary"`
- **Multiclass:** `task="multiclass", num_classes=X` plus choose an averaging strategy

### Averaging Strategies Explained

To understand the difference, imagine a **3-class animal classifier** evaluated on 150 images:  

```
Dataset:
├─ Dog: 100 images (large class)
├─ Cat: 40 images (medium class)
└─ Bird: 10 images (small class, rare)
```

After evaluation, per-class F1 scores:

```
Dog F1 = 0.90 (well represented in dataset)
Cat F1 = 0.70 (less data, harder)
Bird F1 = 0.40 (very few samples, model struggles)
```

Now we need **one number** to describe overall performance. The three strategies give very different answers:
  
==**Macro-average** → Simple mean, all classes treated equally==

```
Macro F1 = (0.90 + 0.70 + 0.40) / 3 = 0.67

Dog: ██████████ 0.90 weight = 1/3
Cat: ███████ 0.70 weight = 1/3
Bird: ████ 0.40 weight = 1/3
─────────────

average = 0.67
```

==✅ Use when: Every class matters equally regardless of frequency==
⚠️ Sensitive to poor performance on rare classes (Bird drags the score down)

==**Weighted-average** → Mean weighted by number of samples per class==

```
Weighted F1 = (100×0.90 + 40×0.70 + 10×0.40) / 150
= (90 + 28 + 4) / 150
= 0.81

Dog: ██████████ 0.90 weight = 100/150 = 67%
Cat: ███████ 0.70 weight = 40/150 = 27%
Bird: ████ 0.40 weight = 10/150 = 7%
─────────────

average = 0.81
```

==✅ Use when: Class frequency in the dataset reflects real-world distribution==
⚠️ Can hide poor performance on rare but important classes (Bird barely matters)

==**Micro-average** → Pool all TP, FP, FN across classes, then compute once==

```
Imagine totals across all classes:

Total TP = 90 + 28 + 4 = 122
Total FP = 10 + 12 + 6 = 28
Total FN = 10 + 12 + 6 = 28

Micro Precision = 122 / (122 + 28) = 0.81
Micro Recall = 122 / (122 + 28) = 0.81
Micro F1 = 0.81

(In multiclass, micro F1 ≈ accuracy)
```

==✅ Use when: You care about total correct predictions across everything==
==⚠️ Dominated by large classes — Bird is nearly invisible==

The course uses `average="macro"` as the default, a safe choice that doesn't hide poor performance on any single class.

**Typical workflow:**

```python
import torchmetrics

# Create metrics
f1 = torchmetrics.F1Score(task="multiclass", num_classes=10, average="macro")

# During evaluation loop
for images, labels in val_loader:
	predictions = model(images)
	f1.update(predictions, labels) # Accumulate

# Get final score
print(f"F1 Score: {f1.compute():.3f}")
```

The metric is updated on each batch and computed at epoch end to evaluate overall performance.

To see an example of which metric to choose in a real scenario see [[Real World Examples Metrics]]

# 3. Learning Rate Schedulers

### What is Optimization?
  
**Optimization** in ML means tuning your model to achieve the best possible performance according to a specific metric like precision, recall, etc... 

==A **hyperparameter** is a setting that controls how a model is trained== and is chosen before or during training rather than learned directly from the data. The **learning rate** is one of the most fundamental hyperparameters, controlling how much the model's parameters are updated at each training step.
### The Learning Rate Effect
  
The learning rate determines the step size the optimizer takes when updating weights each iteration. Its relationship with accuracy follows a clear pattern:

![Learning Rate vs Accuracy](images/lr_vs_accuracy.png)
  
- **Too small** → training is slow, model stagnates at a suboptimal point
- **Too large** → model bounces around, never converges
- **Just right** → highest accuracy

This forms an **inverted U-shape**: performance is low at both extremes, peaks in the middle. The goal of optimization is to find that peak.

### Before Tuning: How can you improve these metrics?
  
==Before touching any hyperparameter, rule out **data problems**. The model can only be as good as what you feed it.==

```
EXTERNAL FACTORS (check these first)
──────────────────────────────────────────────────────────
Data quantity      Not enough examples of a class?
                   → Model won't learn to recognize it
                   → Gather more data from open datasets

Data quality       Noisy labels?
                   → Model learns wrong patterns
                   → "Garbage in, garbage out"

Feature quality    Blurry or low-resolution images?
                   → Model can't find useful patterns
                   → Apply preprocessing: resize, crop, normalize
```
  
```
INTERNAL FACTORS
──────────────────────────────────────────────────────────
Architecture       Number of layers, neurons per layer,
                   activation functions

Regularization     Dropout rate, weight decay

Training           Learning rate, batch size,
                   number of epochs, optimizer choice
```

## Learning Rate Schedulers

A fixed learning rate always forces a trade-off between speed and precision:

![High vs Low Learning Rate|630](images/high_vs_low_lr.png)

- ==**High LR:** Accuracy rises fast in early epochs, then flatlines==
- ==**Low LR:** Climbs slowly but eventually surpasses the high LR, at the cost of many more epochs==

**The question:** Can we get the fast start of a high LR *and* the precision of a low LR?
**The answer:** Yes — with a **learning rate scheduler**.

![[Learning Rate Schedulers]]

# 4. Tuning Hyperparameters

