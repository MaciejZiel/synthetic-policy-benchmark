# LLM Formula Reverse-Engineering Benchmark

A benchmark task that evaluates an LLM's ability to reverse-engineer a hidden
mathematical formula from data. The model receives 100 labeled rows with 16
feature columns and must discover the exact scoring function.

## The Problem

The hidden formula uses **8 out of 16 columns** with trigonometric functions,
conditionals, and a floor term. The challenge: trig terms oscillate many times
across the variable range, producing near-zero linear correlation -- so
regression captures at best ~50% of the variance.

A human expert can solve this in ~1-2 hours using visual residual analysis
and FFT. LLMs cannot see plots and default to regression, making this a
natural test of analytical methodology.

## Results (6 Models x 10 Runs)

| Model | Avg Score | Best | Worst | Zero-score Runs |
|---|---|---|---|---|
| **GPT-5.2** | **0.49** | 0.59 | 0.35 | 0 |
| Claude Sonnet 4.5 | 0.37 | 0.50 | 0.04 | 0 |
| Gemini 2.5 Pro | 0.27 | 0.44 | 0.18 | 0 |
| Llama 4 Maverick | 0.25 | 0.49 | 0.00 | 3 |
| DeepSeek R1 | 0.21 | 0.56 | 0.00 | 5 |
| Qwen 3 235B | 0.16 | 0.40 | 0.00 | 3 |
| **Reference (exact)** | **1.00** | -- | -- | -- |

Across 60 total runs, no model discovered the true trigonometric structure.
Best scores come from simple linear regression, not formula discovery.

See [MODEL_EVALUATION.md](MODEL_EVALUATION.md) for full per-model analysis
and [SOLUTION.md](SOLUTION.md) for the human expert walkthrough.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset (already included)
python src/generate_dataset.py

# Run benchmark against a model
cp .env.example .env
# Edit .env with your OpenRouter API key
python src/run_benchmark.py

# Score manually (auto-scoring is built into run_benchmark.py)
python src/score_predictions.py <model_slug>
```

## Project Structure

```
dataset/
  history_signals.csv       # 100 rows with score (given to LLM)
  target_signals.csv        # 20 rows without score
artifacts/
  hidden_test.csv           # 500 rows for final scoring (never shown)
  reference_answer.json     # Ground truth for target rows
  model_runs/               # Output from benchmark runs
src/
  generate_dataset.py       # Dataset generator with hidden formula
  reference_solution.py     # Ground truth predict() function
  run_benchmark.py          # LLM benchmark runner (OpenRouter API)
  score_predictions.py      # Scoring against hidden test set
task_instructions.txt       # Prompt given to the LLM
scoring_definition.md       # Metric definition
SOLUTION.md                 # Full solution walkthrough
MODEL_EVALUATION.md         # Detailed evaluation report (60 runs)
```

## How It Works

1. **Dataset**: 16 feature columns (x1-x16), only 8 used in the formula
2. **Formula**: 7 terms -- trigonometric functions, conditionals, floor, offset
3. **LLM gets**: 100 rows + hints about operation types + 5 conversation turns with code execution
4. **Scoring**: `score = 1 - clamp(MAE / P85_P15, 0, 1)` on 500 hidden rows

## Testing

```bash
pytest tests/ -v
```

41 tests covering: formula correctness, scoring metric, dataset integrity,
code extraction, reference solution consistency.

## Configuration

Environment variables (`.env`):
- `OPENROUTER_API_KEY` -- API key for OpenRouter
- `MODEL_NAME` -- model identifier (default: `openai/gpt-5.2`)
- `MAX_CALLS` -- number of conversation turns (default: `5`)
