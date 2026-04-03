# Scoring Definition

## Method

The model must output a Python function `def predict(x1, x2, ..., x16)`
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

The model sees 100 rows to reverse-engineer the formula. A regression fit
captures only partial variance (OLS baseline scores ~0.56) because it cannot
represent the non-linear structure:

- Trigonometric terms oscillate many times across the variable range,
  producing near-zero linear correlation with individual features
- Piecewise conditionals create sharp thresholds that polynomials cannot
  approximate without overfitting
- Step functions (`floor`) with non-obvious divisors add discrete jumps
- Feature selection among 16 columns (only a subset matters) adds noise
  that degrades regression further

A regression-based answer will plateau around 0.5. Only discovering the
actual mathematical structure pushes the score toward 1.0.

## Interpretation

| Score | Meaning |
|---|---|
| 1.0 | Perfect - exact formula discovered |
| 0.7-0.9 | Most terms correct, minor coefficient errors |
| 0.4-0.7 | Partial - some structure found, key terms missed |
| 0.1-0.4 | Weak - rough approximation only |
| 0.0 | Complete failure |
