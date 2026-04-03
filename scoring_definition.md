# Scoring Definition

## Method

The model must output a Python function `def predict(x1, x2, x3, x4, x5, x6, x7, x8)`
that computes a score from the features. This function is evaluated on a **hidden
test set of 500 rows** that the model never saw.

## Metric

    score = 1 - clamp(MAE / P85_P15, 0, 1)

- **MAE** = mean absolute error on 500 hidden test rows.
- **P85_P15** = 85th percentile minus 15th percentile of true scores in the
  hidden test set. This robust measure of spread ignores extreme outliers
  while capturing the main distribution of scores.

Result is in **[0, 1]** where 1.0 = perfect match.

## Why this works

The model sees 100 rows to reverse-engineer the formula. A regression fit may
look good on training data but diverges on 500 unseen rows because it cannot
capture:

- Trigonometric terms with multi-variable arguments (`sin(ax + by + cz)`)
- Non-linear functions of variable interactions (`sqrt(x*y - z)`)
- Piecewise conditionals with specific thresholds
- Step functions (`floor`) with non-obvious divisors

Only the exact (or near-exact) formula scores well on the hidden test set.

## Interpretation

| Score | Meaning |
|---|---|
| 1.0 | Perfect - exact formula discovered |
| 0.7-0.9 | Most terms correct, minor coefficient errors |
| 0.4-0.7 | Partial - some structure found, key terms missed |
| 0.1-0.4 | Weak - rough approximation only |
| 0.0 | Complete failure |
