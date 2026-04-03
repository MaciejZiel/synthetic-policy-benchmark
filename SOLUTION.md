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
analytical techniques. Here is the step-by-step approach:

### Step 1: Feature Selection

```python
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/history_signals.csv")
features = [f"x{i}" for i in range(1, 17)]

# Check correlations — most will be near zero
corr = df[features].corrwith(df["score"]).abs().sort_values(ascending=False)
print(corr)
# x4 shows weak correlation (~0.15) from the threshold term
# Others near zero — trig terms destroy linear correlation

# Mutual information is more informative for non-linear relationships
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(df[features], df["score"], random_state=42)
mi_series = pd.Series(mi, index=features).sort_values(ascending=False)
print(mi_series)
# x3, x9 will show highest MI (dominant trig terms)
# x6, x14, x4, x7, x16, x11 also elevated
# x1, x2, x5, x8, x10, x12, x13, x15 near zero — noise columns
```

**Result**: Identifies ~8 relevant columns, eliminates 8 noise columns.

### Step 2: Discover sin(0.5 * x3)

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Fit a basic linear model to get residuals
X = df[features].values
model = LinearRegression().fit(X, df["score"])
residuals = df["score"] - model.predict(X)

# Plot residuals vs x3
plt.scatter(df["x3"], residuals, alpha=0.5)
plt.xlabel("x3"); plt.ylabel("Residual"); plt.title("Residuals vs x3")
plt.show()
# Clear sinusoidal pattern visible!

# FFT to find frequency
from scipy.fft import fft, fftfreq
# Sort by x3 for FFT
sorted_idx = df["x3"].argsort()
x3_sorted = df["x3"].values[sorted_idx]
res_sorted = residuals.values[sorted_idx]

# Interpolate to uniform grid for FFT
from scipy.interpolate import interp1d
x3_uniform = np.linspace(x3_sorted.min(), x3_sorted.max(), 256)
f_interp = interp1d(x3_sorted, res_sorted, kind="linear", fill_value="extrapolate")
res_uniform = f_interp(x3_uniform)

freqs = fftfreq(256, d=(x3_uniform[1] - x3_uniform[0]))
power = np.abs(fft(res_uniform))
# Peak at frequency ~0.08 cycles/unit → angular frequency = 2*pi*0.08 ≈ 0.5
# Amplitude from power spectrum ≈ 35
```

**Result**: `35.0 * sin(0.5 * x3)` — first term discovered.

### Step 3: Subtract and find cos(0.4 * x9)

```python
# Remove discovered term
df["residual2"] = df["score"] - 35.0 * np.sin(0.5 * df["x3"])

# Plot residual2 vs x9
plt.scatter(df["x9"], df["residual2"], alpha=0.5)
plt.show()
# Cosine pattern visible!

# Same FFT approach → frequency ≈ 0.4, amplitude ≈ 28, phase → cos
```

**Result**: `28.0 * cos(0.4 * x9)` — second term discovered.

### Step 4: Subtract and find sin(0.3*x6 - 0.2*x14)

```python
df["residual3"] = df["residual2"] - 28.0 * np.cos(0.4 * df["x9"])

# Plot vs each remaining variable — x6 shows noisy sinusoidal pattern
# The noise comes from x14 dependency
# Test: color scatter plot by x14 → pattern clarifies
plt.scatter(df["x6"], df["residual3"], c=df["x14"], cmap="viridis", alpha=0.7)
plt.colorbar(label="x14")
plt.show()

# Try linear combination: 0.3*x6 - 0.2*x14
combo = 0.3 * df["x6"] - 0.2 * df["x14"]
plt.scatter(combo, df["residual3"], alpha=0.5)
plt.show()
# Clean sine wave visible! Amplitude ≈ 20
```

**Result**: `20.0 * sin(0.3 * x6 - 0.2 * x14)` — hardest term, found via
pairwise exploration. This is the step that requires the most expertise.

### Step 5: Identify thresholds from remaining residuals

```python
df["residual4"] = df["residual3"] - 20.0 * np.sin(0.3 * df["x6"] - 0.2 * df["x14"])

# Sort by x4 → see jump at x4 = 25
df_sorted = df.sort_values("x4")
plt.plot(df_sorted["x4"].values, df_sorted["residual4"].values, "o-", alpha=0.3)
plt.show()
# Clear step: ~+5 above x4=25, ~-5 below

# After removing x4 threshold:
df["residual5"] = df["residual4"] - np.where(df["x4"] > 25, 5.0, -5.0)

# Try sums of variable pairs → x7 + x16 shows jump at 50
df["x7_x16"] = df["x7"] + df["x16"]
df_sorted = df.sort_values("x7_x16")
plt.plot(df_sorted["x7_x16"].values, df_sorted["residual5"].values, "o-", alpha=0.3)
plt.show()
# Step at 50: -4 above, +4 below
```

**Result**: Two threshold terms discovered.

### Step 6: Identify floor term and offset

```python
df["residual6"] = df["residual5"] - np.where(df["x7"] + df["x16"] > 50, -4.0, 4.0)

# Plot vs x11 → staircase pattern with steps at multiples of 7
plt.scatter(df["x11"], df["residual6"], alpha=0.5)
plt.show()
# Steps at x11 = 7, 14, 21, 28, 35, 42 → floor(x11 / 7) * 2

# Final residual after all terms → constant ≈ 50
df["residual7"] = df["residual6"] - 2.0 * np.floor(df["x11"] / 7.0)
print(df["residual7"].describe())
# mean ≈ 50, std ≈ 0 → offset = 50
```

**Result**: Floor term + offset. Formula complete.

### Verification

```python
import math

def predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16):
    return round(
        35.0 * math.sin(0.5 * x3)
        + 28.0 * math.cos(0.4 * x9)
        + 20.0 * math.sin(0.3 * x6 - 0.2 * x14)
        + (5.0 if x4 > 25 else -5.0)
        + (-4.0 if x7 + x16 > 50 else 4.0)
        + 2.0 * math.floor(x11 / 7.0)
        + 50
    , 2)

# Test on training data
errors = [abs(row["score"] - predict(**{f"x{i}": row[f"x{i}"] for i in range(1,17)}))
          for _, row in df.iterrows()]
print(f"Max error: {max(errors)}")  # 0.0 — perfect match
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
