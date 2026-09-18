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