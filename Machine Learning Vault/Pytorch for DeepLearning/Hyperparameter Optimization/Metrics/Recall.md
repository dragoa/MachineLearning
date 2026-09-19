##### Definition
==How often the model identifies true positives from all positives==

$$

\begin{aligned}

\text{Recall} &= \frac{TP}{TP + FN}

\end{aligned}

$$

- **Interpretation:** Did we find all the positive cases?
- Goal: maximize, as close to 1 as possible.
- **Use when:** ==False negatives are costly== (we want to minimize false negatives)
	- Cancer screening (missed diagnosis = death)
	- Fraud detection (missed fraud = money loss)
	- Security alerts (missed threat = breach)

**Example:** 100 actual cancer cases exist
- Model catches 85 → Recall = 85% (15 missed diagnoses)