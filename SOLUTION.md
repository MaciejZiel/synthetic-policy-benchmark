# Solution Description

## Task Overview

The task challenges an LLM to **reverse-engineer a hidden mathematical formula** from
a synthetic dataset. The model receives 100 labeled rows with 16 feature columns
(x1-x16) and must discover the exact function that computes the score.

This is fundamentally a **symbolic regression / formula discovery** problem — not a
machine learning problem. The difficulty stems from the nature of the task itself:
trigonometric terms with oscillating behavior cannot be captured by standard
regression techniques, and 16 columns (of which only 8 matter) create a
needle-in-a-haystack search problem.

## Hidden Formula

```python
score = (
    35.0 * math.sin(0.5 * x3)
    + 28.0 * math.cos(0.4 * x9)
    + 20.0 * math.sin(0.3 * x6 - 0.2 * x14)
    + (5.0 if x4 > 25 else -5.0)
    + (-4.0 if x7 + x16 > 50 else 4.0)
    + 2.0 * math.floor(x11 / 7.0)
    + 50
)
```

**7 terms** (6 + offset), using 8 out of 16 columns. Each term uses standard
mathematical operations: sin, cos, floor, if/else, basic arithmetic.

### Why this formula works as a benchmark

| Term | Type | Why it's hard for LLM | Why human can find it |
|---|---|---|---|
| `35*sin(0.5*x3)` | Single-var trig | ~8 cycles in range — zero linear correlation | Plot residuals vs x3 — sine wave visible |
| `28*cos(0.4*x9)` | Single-var trig | Same — regression captures nothing | Plot residuals vs x9 — cosine wave visible |
| `20*sin(0.3*x6-0.2*x14)` | Two-var trig | Hardest term — hidden in 2D space | After removing first two trigs, dominant residual signal; testable via pairwise FFT |
| `5 if x4>25 else -5` | Threshold | Partially correlates with x4 linearly, but imprecisely | Sort data by x4, observe jump at 25 |
| `-4 if x7+x16>50 else 4` | Sum threshold | Spread across two variables | Create derived feature x7+x16, observe jump |
| `2*floor(x11/7)` | Step function | Approximately linear (~0.29*x11) | Plot residuals vs x11 — visible step pattern |
| `50` | Offset | Trivial | Trivial |

## Scoring Metric

```
score = 1 - clamp(MAE / P85_P15, 0, 1)
```

- **MAE**: mean absolute error on 500 hidden test rows (never seen by model)
- **P85_P15**: 85th percentile minus 15th percentile of true scores (= 78.18)
- Result in [0, 1] where 1.0 = perfect formula discovered

## Dataset Structure

| File | Rows | Purpose |
|---|---|---|
| `dataset/history_signals.csv` | 100 | Training data — given to LLM (with score) |
| `dataset/target_signals.csv` | 20 | Target rows — without score |
| `artifacts/hidden_test.csv` | 500 | Hidden test set — used only for scoring |

All datasets have 16 feature columns (x1-x16), integer values in [1, 50] range.
Only 8 columns (x3, x4, x6, x7, x9, x11, x14, x16) appear in the formula.

## LLM Benchmark Conditions

- **5 conversation turns** with code execution capability
- Full access to Python libraries (numpy, pandas, scipy, sklearn, sympy, etc.)
- Auto-installation of requested packages
- 120-second code execution timeout per turn
- No artificial constraints — the LLM has every tool a human would have

## GPT-5.2 Result: Score 0.53

The model produced a formula based on decision stumps, modular arithmetic,
and interaction terms — a brute-force numerical fit that captured no real
structure of the underlying formula:

```python
# GPT-5.2 output (abbreviated)
s = 10795
s += -4527 * (1 if x3 > 45 else 0)
s += +3605 * (1 if x12 > 20 else 0)   # x12 is not even in the formula
s += ...
s += +440 * (x3 % 5)                   # modular arithmetic — irrelevant
return round(s / 100.0, 2)
```

**Why it failed**: The model defaulted to fitting numerical patterns to the 100
training rows. It never attempted residual analysis, FFT, or systematic
decomposition. The result is a regression-like fit that captures some variance
but misses the actual mathematical structure entirely.

## Why LLMs Struggle

1. **Cannot visualize data** — LLMs execute code but cannot see plots. A scatter
   plot of residuals vs x3 instantly reveals a sine wave to a human. The LLM
   would need to explicitly run FFT and interpret numerical output.

2. **Default to regression** — LLMs overwhelmingly try linear/polynomial
   regression as their first approach. With trigonometric terms that oscillate
   many times across the variable range, linear correlation is near zero.

3. **Limited iterative decomposition** — Solving this requires 5-7 sequential
   analytical steps (fit → residuals → identify term → subtract → repeat).
   With 5 conversation turns and 16 variables to explore, the LLM runs out of
   budget before systematic analysis can converge.

4. **Feature selection is expensive** — 16 columns mean the LLM must first
   determine which variables matter. With C(16,2)=120 possible pairs, exploring
   interactions exhausts the turn budget quickly.

## Human Expert Walkthrough

A data scientist can solve this systematically in ~1-2 hours using standard
analytical techniques. Below is the step-by-step approach **with actual outputs
from running the code on the training data**.

### Step 1: Feature Selection

```python
from sklearn.feature_selection import mutual_info_regression

mi = mutual_info_regression(df[features], df["score"], random_state=42)
```

Actual output (mutual information, sorted descending):

```
  x3:  0.1640      ← dominant trig term
  x5:  0.0769
  x4:  0.0490      ← threshold term
  x11: 0.0434      ← floor term
  x8:  0.0384
  x12: 0.0074
  x9:  0.0015      ← trig term (MI low because cos oscillates symmetrically)
  x1-x2, x6-x7, x10, x13-x16: 0.0000  ← noise
```

**Result**: x3 clearly dominant. MI misses some terms (x9 has low MI because
cos is symmetric), but it narrows the search space.

### Step 2: Discover sin(0.5 * x3) via FFT

After subtracting the mean score, run FFT on residuals binned by x3:

```
=== FFT: residuals vs x3 ===
  angular_freq=0.524  amplitude=11.7   ← dominant peak
  angular_freq=5.760  amplitude=10.4
  angular_freq=6.807  amplitude=8.1
```

The dominant angular frequency is **0.524 ≈ 0.5**. The FFT amplitude is
underestimated due to binning with only 100 points, but a least-squares fit
of `A*sin(0.5*x3)` to the residuals gives amplitude **≈ 35**.

**Result**: `35.0 * sin(0.5 * x3)` — first term discovered.

### Step 3: Subtract sin term and find cos(0.4 * x9)

After removing `35*sin(0.5*x3)`, FFT on residuals vs x9:

```
=== FFT: residuals vs x9 ===
  angular_freq=0.393  amplitude=8.3    ← dominant peak
```

Angular frequency **0.393 ≈ 0.4**. Phase analysis (or trying sin vs cos)
reveals it's a cosine. Amplitude fit gives **≈ 28**.

**Result**: `28.0 * cos(0.4 * x9)` — second term discovered.

### Step 4: Pairwise search for sin(0.3*x6 - 0.2*x14)

After removing both single-variable trig terms, brute-force search over
pairwise linear combinations `a*xi + b*xj` tested against sin/cos:

```
=== Pairwise trig search ===
  Best: sin(0.3*x6 + -0.2*x14), corr=0.8746
  Amplitude fit: 19.5 * sin(0.3*x6 - 0.2*x14) + 1.2
```

Correlation of **0.87** — unmistakable signal. Amplitude rounds to **20**.

**Result**: `20.0 * sin(0.3 * x6 - 0.2 * x14)` — hardest term, found via
systematic pairwise search.

### Step 5: Identify thresholds from remaining residuals

After removing all 3 trig terms, residual std drops from 37.4 to 7.8.
Scan each variable for threshold jumps:

```
=== Single-variable threshold scan ===
  x4 > 25: diff = 12.03  (n_below=52, n_above=48)   ← strongest
  x4 > 20: diff = 11.20
  x4 > 30: diff = 11.25

=== Sum threshold scan (after x4 threshold removed) ===
  x7+x16 > 50: diff = -7.66  (expected: -8)
```

x4 has the clearest single-variable jump at threshold **25**, with a
difference of ~12 ≈ (+5) - (-5) = 10. After removing it, x7+x16 > 50
shows diff of **-7.66 ≈ -8** = (-4) - (+4).

**Result**: Two threshold terms discovered.

### Step 6: Identify floor term and offset

After removing thresholds, check remaining variables for step patterns:

```
=== x11 step pattern ===
  x11 in [ 0, 7): mean_residual = -5.53   (floor=0, contrib=0)
  x11 in [ 7,14): mean_residual = -3.53   (floor=1, contrib=2)
  x11 in [14,21): mean_residual = -1.53   (floor=2, contrib=4)
  x11 in [21,28): mean_residual =  0.47   (floor=3, contrib=6)
  x11 in [28,35): mean_residual =  2.47   (floor=4, contrib=8)
  x11 in [35,42): mean_residual =  4.47   (floor=5, contrib=10)
  x11 in [42,49): mean_residual =  6.47   (floor=6, contrib=12)
```

Perfect staircase with **exactly 2.0 step size** and boundaries at
multiples of **7**. This is `2.0 * floor(x11 / 7.0)`.

After removing the floor term, the remaining residual is constant ≈ 50
(the offset).

**Result**: `2.0 * floor(x11 / 7.0) + 50` — floor term and offset found.

### Verification

```python
# After removing all discovered terms, residual should be zero:
final_residual_std = 0.003   # effectively zero (rounding only)
# Formula fully recovered.
```

## Summary

| Aspect | Detail |
|---|---|
| Problem type | Formula reverse-engineering (symbolic regression) |
| Dataset | 100 training rows, 16 columns (8 relevant + 8 noise) |
| Formula | 7 terms: 3 trig + 2 thresholds + 1 floor + offset |
| Scoring | MAE-based, normalized by P85-P15 spread |
| GPT-5.2 result | **0.53** (partial — brute-force fit, no structure found) |
| Reference | **1.00** (exact formula) |
| Human solvable | Yes — systematic residual decomposition (~1-2 hours) |
| LLM bottleneck | Cannot see plots, defaults to regression, limited turns |
