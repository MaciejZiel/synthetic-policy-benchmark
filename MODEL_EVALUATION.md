# Model Evaluation Report

Evaluation of 6 frontier LLMs on the formula reverse-engineering benchmark.
Each model was given **10 independent runs** (60 total) to control for variance.

## Models Tested

| Model | Type | Provider |
|---|---|---|
| GPT-5.2 | General | OpenAI |
| Claude Sonnet 4.5 | General | Anthropic |
| Gemini 2.5 Pro | General | Google |
| Llama 4 Maverick | Open-weight | Meta |
| Qwen 3 235B (A22B) | Open-weight (MoE) | Alibaba |
| DeepSeek R1 | Reasoning | DeepSeek |

All models accessed via OpenRouter API. Each run: 5 conversation turns with
Python code execution, auto-install of common libraries (numpy, pandas, scipy,
sklearn, sympy, matplotlib), 120s execution timeout per turn.

## Results

### Scores by Run

| Model | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-5.2 | 0.53 | 0.53 | 0.52 | 0.35 | 0.47 | **0.59** | 0.58 | 0.38 | 0.51 | 0.48 |
| Claude Sonnet 4.5 | 0.50 | -- | 0.46 | 0.06 | 0.47 | 0.37 | 0.48 | 0.48 | 0.45 | 0.04 |
| Gemini 2.5 Pro | 0.22 | 0.22 | 0.28 | 0.31 | 0.44 | 0.34 | 0.24 | 0.18 | 0.18 | 0.25 |
| Llama 4 Maverick | 0.28 | 0.00 | 0.29 | 0.38 | 0.49 | 0.31 | 0.00 | 0.00 | 0.28 | 0.42 |
| Qwen 3 235B | 0.00 | 0.00 | 0.14 | 0.14 | 0.40 | 0.18 | 0.34 | 0.33 | 0.03 | 0.00 |
| DeepSeek R1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.46 | 0.00 | 0.20 | 0.50 | 0.38 | 0.56 |

Claude Sonnet R2 marked `--`: failed to produce a valid function (execution error, not scoring failure).

### Summary Statistics

| Model | Avg | Median | Best | Worst | Std Dev | Zero-score runs |
|---|---|---|---|---|---|---|
| **GPT-5.2** | **0.49** | **0.51** | 0.59 | 0.35 | 0.08 | 0 |
| Claude Sonnet 4.5 | 0.37 | 0.46 | 0.50 | 0.04 | 0.17 | 0 |
| Gemini 2.5 Pro | 0.27 | 0.24 | 0.44 | 0.18 | 0.08 | 0 |
| Llama 4 Maverick | 0.25 | 0.28 | 0.49 | 0.00 | 0.17 | 3 |
| DeepSeek R1 | 0.21 | 0.10 | 0.56 | 0.00 | 0.23 | 5 |
| Qwen 3 235B | 0.16 | 0.14 | 0.40 | 0.00 | 0.15 | 3 |

Reference (exact formula): **1.00**. Human expert: **1.00** (see [SOLUTION.md](SOLUTION.md)).

## Per-Model Analysis

### GPT-5.2 -- Best Average (0.49)

The most consistent performer. Defaults to OLS linear regression in most runs,
which reliably scores ~0.52. When it deviates from OLS, results are mixed:

- **Best run (R6, 0.59)**: Produced the simplest formula of any model across
  all 60 runs -- `0.92 * x12 + 34.68`. A single linear term. GPT explicitly
  stated "cannot uniquely recover the true formula from 100 rows" and opted for
  the best simple approximation. Paradoxically, the least ambitious approach
  yielded the highest score.
- **Worst run (R4, 0.35)**: Attempted a decision tree with conditional thresholds
  and a modular arithmetic "cents" component. Overengineered and underperformed.
- Never attempts trigonometric discovery. Never identifies correct variables.

### Claude Sonnet 4.5 -- High Ceiling, Unstable (0.37)

Consistently builds formulas around `x2 + x10 + x13` as base features (all
incorrect, but correlated enough with the target to sustain ~0.45). Adds
trigonometric terms with wrong variables and frequencies that accidentally
provide partial signal.

- **Best runs (R1/R7/R8, ~0.48-0.50)**: Mixed linear + trig approach.
- **Catastrophic runs (R4=0.06, R10=0.04)**: Same strategy, wrong trig
  combinations. Minor formula changes cause major score swings.
- **R2 failure**: Sent multiple code blocks per message. Only the first executes
  in the sandbox; the second referenced undefined variables. Never recovered.
- Problem: doesn't understand isolated subprocess execution model.

### Gemini 2.5 Pro -- Most Deterministic (0.27)

Runs 1-2 produced identical scores (0.22) with identical MAE (60.74) --
the most deterministic model tested. Later runs showed more variance.

- Consistently selects x10 as a prominent feature (incorrect).
- **Best run (R5, 0.44)**: Clean combination of linear, threshold, and
  sin/cos terms.
- **Run 9**: Used `math.radians()` to convert inputs before trig --
  the same error pattern seen in DeepSeek.
- Run 7 was unique: split the formula into two entirely different regimes
  via if/else on x2.

### Llama 4 Maverick -- Extreme Variance (0.25)

The most unpredictable model. Three runs scored 0.00 while one hit 0.49.

- **Runs scoring 0.00 (R2, R7, R8)**: All used multiplicative feature
  interactions (e.g., `x9*(x1+x3)`, `x3*x9`, `x8*sin(x1)`).
  Product terms are the single most reliable predictor of total failure
  across all models.
- **Runs scoring >0.3 (R4, R5, R10)**: All used conditional/piecewise logic
  with thresholds. This approach consistently outperforms multiplication.
- Sometimes accidentally uses relevant variables (x9, x6) but with wrong
  structure.

### Qwen 3 235B -- Improving but Fragile (0.16)

Started terribly (0.00 in R1 and R2) but improved in later runs.

- **Best run (R5, 0.40)**: Simple weighted linear combination of 5 features
  plus two conditional terms. The simplest formula produced the best result.
- **Worst run (R10, 0.00, MAE=347)**: Pairwise product ratios plus
  trigonometric transforms. The most complex formula produced the worst result.
  This is the clearest single demonstration of the complexity-failure pattern.
- Run 7 (0.34): Pure linear combination of 8 features, no nonlinearities.
  Outperformed every run that attempted sophisticated approaches.

### DeepSeek R1 -- Reasoning Model, Biggest Surprise (0.21)

The most dramatic trajectory. First 4 runs all scored **0.00**, then
suddenly produced the single highest score across all models (0.56).

- **Runs 1-4 (all 0.00)**: Persistent `math.radians()` bug -- converting
  inputs to radians (dividing by ~57) before applying trig functions,
  effectively making all trig terms nearly linear. Extended thinking
  didn't catch this error across 4 consecutive runs.
- **Run 10 (0.56, best overall)**: Used `10*floor(x13/10)` as a dominant
  feature plus sin/cos and a conditional. While the variables are wrong (x13
  instead of x11), the approach of using floor as a dominant predictor was
  the most structurally insightful of any model.
- **Run 6 (0.00, MAE=375)**: All pairwise quadratic interactions plus generic
  trig expansion. MAE of 375 is the worst single prediction across all 60 runs.
- Extended reasoning is double-edged: it can produce the best result (0.56) or
  the worst (MAE=375). More thinking amplifies both insight and overconfidence.

## Key Findings

### 1. No Model Discovers the True Formula Structure

Across 60 runs, **zero** produced `sin(0.5*x3)` or `cos(0.4*x9)` with
correct variables and frequencies. The best single run (0.59) still misses
41% of the variance. Every model achieves its scores through approximate
numerical fits, not structural discovery.

### 2. Simplicity Outperforms Complexity

This pattern holds across every model without exception:

| Approach type | Typical score range | Example |
|---|---|---|
| Single linear term | 0.50-0.59 | GPT R6: `0.92*x12 + 34.68` |
| OLS regression | 0.47-0.53 | GPT R1-R3 |
| Simple conditionals | 0.28-0.49 | Llama R5, Qwen R5 |
| Trig with wrong vars | 0.18-0.48 | Claude, Gemini |
| Product interactions | 0.00 | Llama R7/R8, Qwen R10 |
| Polynomial expansion | 0.00 | DeepSeek R6 |

Models that try to be "smart" almost always score worse than models that
default to simple regression. The paradox: being more ambitious leads to
worse results because the models can't identify the right structure.

### 3. Feature Selection Is Effectively Random

The true formula uses x3, x4, x6, x7, x9, x11, x14, x16. Models
consistently select wrong variables:

- GPT-5.2 favors x12 (noise variable -- yet scores highest on it)
- Claude Sonnet anchors on x2, x10, x13 (all noise)
- Gemini consistently selects x10 (noise)
- No model across 60 runs correctly identified the full set of 8 relevant variables

### 4. Stability Varies by an Order of Magnitude

| Model | Score range | Spread |
|---|---|---|
| GPT-5.2 | 0.35 -- 0.59 | 0.24 |
| Gemini 2.5 Pro | 0.18 -- 0.44 | 0.26 |
| Claude Sonnet 4.5 | 0.04 -- 0.50 | 0.46 |
| Qwen 3 235B | 0.00 -- 0.40 | 0.40 |
| Llama 4 Maverick | 0.00 -- 0.49 | 0.49 |
| DeepSeek R1 | 0.00 -- 0.56 | 0.56 |

DeepSeek R1 has the widest spread (0.56) -- the reasoning model swings between
brilliance and catastrophe. GPT-5.2 is the most reliable with the narrowest
spread (0.24) and no zero-score runs.

### 5. Extended Reasoning Doesn't Help (On Average)

DeepSeek R1, the only reasoning model tested, scored worst on average (0.21).
It had the most zero-score runs (5/10) AND the single best run (0.56). Extended
chain-of-thought reasoning amplifies variance rather than improving accuracy.
The model "overthinks" into overengineered solutions as often as it reasons into
good ones.

### 6. The Human-LLM Gap is Structural

The gap is not about intelligence or compute -- it's about methodology:

| Capability | Human | LLM |
|---|---|---|
| Plot residuals, see sine wave | Yes | No (can execute code, can't see output) |
| Run FFT, identify frequencies | Systematic | Would need to know to try |
| Iterative decomposition | Natural workflow | Runs out of turns |
| Visual pattern recognition | Instant (scatter plot) | Impossible (text only) |

A human data scientist uses visual-first analysis: plot, observe, hypothesize,
subtract, repeat. LLMs default to regression-first analysis: fit, predict,
overfit, fail. The benchmark exploits this methodological difference.

## Methodology

- **API**: OpenRouter (all models accessed through unified API)
- **Turns per run**: 5 (configurable via `MAX_CALLS`)
- **Code execution**: Sandboxed Python subprocess, 120s timeout
- **Libraries available**: numpy, pandas, scipy, sklearn, sympy, matplotlib
  (auto-installed on first use)
- **Scoring**: `score = 1 - clamp(MAE / P85_P15, 0, 1)` on 500 hidden rows
- **P85_P15**: 78.18 (85th - 15th percentile of true scores)
- **Runs**: 10 per model, 60 total
