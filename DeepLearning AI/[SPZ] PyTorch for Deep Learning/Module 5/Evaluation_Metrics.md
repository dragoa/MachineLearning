# PyTorch – Hyperparameter Optimization (Course 2)

> Notes from *PyTorch for Deep Learning Professional Certificate* · DeepLearning.AI  
> Module 1: Evaluation Metrics & Optimization

---

## Table of Contents

1. [Evaluation Metrics](#1-evaluation-metrics)
2. [Introduction to Optimization](#2-introduction-to-optimization) *(coming soon)*
3. [Learning Rate Schedulers](#3-learning-rate-schedulers) *(coming soon)*
4. [Optimizers](#4-optimizers) *(coming soon)*
5. [Batch Size & Training Dynamics](#5-batch-size--training-dynamics) *(coming soon)*
6. [Early Stopping](#6-early-stopping) *(coming soon)*
7. [Search Strategies (Grid, Random, Optuna)](#7-search-strategies) *(coming soon)*

---

## 1. Evaluation Metrics

### Why Metrics Matter

**Accuracy alone is not enough.** Different metrics tell you different things about your model. The metric you optimize determines what your model gets good at.

**Core principle:** Choose the metric that matches your problem's real-world cost structure.

### Classification Outcomes

All classification metrics are based on **four fundamental outcomes**:

```
                 PREDICTED
                    │
         ┌──────────┴──────────┐
         │                     │
      POSITIVE              NEGATIVE
         │                     │
    ┌────┴──────┐         ┌────┴──────┐
    │           │         │           │
   TRUE       FALSE       FALSE      TRUE
 POSITIVE    POSITIVE    NEGATIVE   NEGATIVE
   (TP)       (FP)        (FN)       (TN)
```

**What these mean:**
- **TP:** Predicted positive, was actually positive ✓
- **FP:** Predicted positive, was actually negative ✗
- **FN:** Predicted negative, was actually positive ✗
- **TN:** Predicted negative, was actually negative ✓

### Binary vs Multiclass

**Binary:** One "positive" class you want to detect vs "negative" (everything else)
- Examples: Spam/Not spam, Disease/Healthy, Fraud/Legitimate

**Multiclass:** Multiple distinct classes (no inherent "positive")
- Examples: Dog/Cat/Bird, Digits 0-9, Weather types
- Calculate metrics per class (treating each as "positive" vs rest), then combine using macro/weighted/micro averaging

---

## The Four Main Metrics

### Accuracy
**Definition:** Percentage of all predictions that were correct

```
Accuracy = (TP + TN) / (All predictions)
```

**When to use:** Balanced datasets only  
**Avoid when:** Classes are imbalanced (e.g., 99% healthy, 1% sick)  
**Problem:** A model that always predicts "healthy" gets 99% accuracy!

---

### Precision
**Definition:** Of all positive predictions, how many were actually correct?

```
Precision = TP / (TP + FP)
            Correct positive predictions / All positive predictions
```

**Interpretation:** How trustworthy is the model when it says "yes"?

**Use when:** False positives are costly
- Spam filter (delete legitimate email = bad)
- Fraud detection (block legitimate customer = bad)
- Medical test (unnecessary treatment = bad)

**Example:** Model flags 100 emails as spam
- 90 are actually spam → Precision = 90% (10 false alarms)

---

### Recall
**Definition:** Of all actual positive cases, how many did the model catch?

```
Recall = TP / (TP + FN)
         True positives caught / All actual positives
```

**Interpretation:** Did we find all the positive cases?

**Use when:** False negatives are costly
- Cancer screening (missed diagnosis = death)
- Fraud detection (missed fraud = money loss)
- Security alerts (missed threat = breach)

**Example:** 100 actual cancer cases exist
- Model catches 85 → Recall = 85% (15 missed diagnoses)

---

### F1 Score
**Definition:** Harmonic mean of precision and recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation:** Balances precision and recall into one score

**Use when:**
- Imbalanced datasets (preferred over accuracy)
- You care equally about precision AND recall
- Requirements unclear

**Why harmonic mean?** Heavily penalizes extreme imbalances
- Precision 100%, Recall 50% → F1 ≈ 67% (not 75%)
- Prevents overfitting to one metric

---

---

## Quick Comparison

| Metric | Best For | Avoid When |
|--------|----------|-----------|
| **Accuracy** | Balanced datasets | Classes imbalanced |
| **Precision** | False positives costly | Must catch all cases |
| **Recall** | False negatives costly | Too many false alarms unacceptable |
| **F1 Score** | Imbalanced dataset | Single metric not important |

---

## Computing Metrics in PyTorch

PyTorch's `torchmetrics` library computes these metrics efficiently during evaluation.

**Key configuration:**
- **Binary:** `task="binary"` (one positive/negative threshold)
- **Multiclass:** `task="multiclass", num_classes=X` + choose an averaging strategy

### Averaging Strategies Explained

To understand the difference, imagine a **3-class animal classifier** evaluated on 150 images:

```
Dataset:
├─ Dog:  100 images  (large class)
├─ Cat:   40 images  (medium class)
└─ Bird:  10 images  (small class, rare)
```

After evaluation, per-class F1 scores:
```
Dog  F1 = 0.90   (well represented in dataset)
Cat  F1 = 0.70   (less data, harder)
Bird F1 = 0.40   (very few samples, model struggles)
```

Now we need **one number** to describe overall performance. The three strategies give very different answers:

---

**Macro-average** → Simple mean, all classes treated equally

```
Macro F1 = (0.90 + 0.70 + 0.40) / 3 = 0.67

Dog:  ██████████ 0.90  weight = 1/3
Cat:  ███████    0.70  weight = 1/3
Bird: ████       0.40  weight = 1/3
                       ─────────────
                       average = 0.67
```

✅ Use when: Every class matters equally regardless of frequency  
⚠️ Sensitive to poor performance on rare classes (Bird drags the score down)

---

**Weighted-average** → Mean weighted by number of samples per class

```
Weighted F1 = (100×0.90 + 40×0.70 + 10×0.40) / 150
            = (90 + 28 + 4) / 150
            = 0.81

Dog:  ██████████ 0.90  weight = 100/150 = 67%
Cat:  ███████    0.70  weight =  40/150 = 27%
Bird: ████       0.40  weight =  10/150 =  7%
                                ─────────────
                                average = 0.81
```

✅ Use when: Class frequency in the dataset reflects real-world distribution  
⚠️ Can hide poor performance on rare but important classes (Bird barely matters)

---

**Micro-average** → Pool all TP, FP, FN across classes, then compute once

```
Imagine totals across all classes:
Total TP = 90 + 28 + 4  = 122
Total FP = 10 + 12 + 6  = 28
Total FN = 10 + 12 + 6  = 28

Micro Precision = 122 / (122 + 28) = 0.81
Micro Recall    = 122 / (122 + 28) = 0.81
Micro F1        = 0.81

(In multiclass, micro F1 ≈ accuracy)
```

✅ Use when: You care about total correct predictions across everything  
⚠️ Dominated by large classes — Bird is nearly invisible

---

**Which to choose?**
```
All classes equally important?
(rare class matters as much as common one)
            │
        ┌───┴───┐
        │       │
       YES      NO
        │       │
        ▼       ▼
      MACRO  WEIGHTED
              (frequency reflects real world)
              or MICRO
              (just want total correct %)
```

The course uses `average="macro"` as the default — a safe choice that doesn't hide poor performance on any single class.

**Typical workflow:**
```python
import torchmetrics

# Create metrics
f1 = torchmetrics.F1Score(task="multiclass", num_classes=10, average="macro")

# During evaluation loop
for images, labels in val_loader:
    predictions = model(images)
    f1.update(predictions, labels)  # Accumulate

# Get final score
print(f"F1 Score: {f1.compute():.3f}")
```

The metric is updated on each batch and computed at epoch end to evaluate overall performance.

---

---

## Choosing Your Metric: Real-World Examples

### Medical Diagnosis
**Priority:** Don't miss cases (minimize false negatives)
```
Choose: RECALL
Reason: Missing a patient = death (very costly)
Accept: Some false alarms (unnecessary tests are acceptable)
```

### Email Spam Filter
**Priority:** Don't delete legitimate emails (minimize false positives)
```
Choose: PRECISION
Reason: Deleting real email = user loses important message
Accept: Some spam slips through (annoying but acceptable)
```

### Fraud Detection
**Priority:** Catch fraud AND avoid false alarms
```
Choose: F1 SCORE (or balance both metrics)
Reason: Dataset heavily imbalanced (1% fraud, 99% legitimate)
Note: Accuracy would be useless (99% baseline by always predicting "legitimate")
```

### Image Classification (10 classes)
**Priority:** Fair evaluation across all classes
```
Choose: F1 SCORE with weighted averaging
Reason: Some classes have more training data than others
Benefit: Accounts for imbalance, prevents over-optimizing popular classes
```

---

## Summary

✅ **Accuracy** – Simple, but only for balanced data  
✅ **Precision** – When false positives hurt  
✅ **Recall** – When false negatives hurt  
✅ **F1 Score** – Balanced metric, works with imbalanced data  

**The key:** Choose the metric that matches your problem's cost structure. Optimize for what matters in the real world.
