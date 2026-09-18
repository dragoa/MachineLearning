##### Definition
==Balances false positives and false negatives==

$$

\begin{aligned}

\text{F1} &= 2 * \frac{Precision * Recall}{Precision + Recall}

\end{aligned}

$$

- **Interpretation:** Balances precision and recall into one score
- Goal: maximize, as close to 1 as possible.
- **Use when:**
	- ==Imbalanced datasets (preferred over accuracy)==
	- You care equally about precision AND recall
	- Requirements unclear

**Why harmonic mean?** Heavily penalizes extreme imbalances

- Precision 100%, Recall 50% → F1 ≈ 67% (not 75%)

- Prevents overfitting to one metric