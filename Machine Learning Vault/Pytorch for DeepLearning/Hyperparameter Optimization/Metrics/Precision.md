##### Definition
==How often the model's positive predictions are actually correct==

$$

\begin{aligned}

\text{Precision} &= \frac{TP}{TP + FP}

\end{aligned}

$$

- **Interpretation:** How trustworthy is the model when it says yes?
- Goal: maximize, as close to 1 as possible.
- **Use when:** ==False positives are costly== (we want to minimize false positives)
	- Spam filter (delete legitimate email = bad)
	- Fraud detection (block legitimate customer = bad)
	- Medical test (unnecessary treatment = bad)

**Example:** Model flags 100 emails as spam
- 90 are actually spam → Precision = 90% (10 false alarms)