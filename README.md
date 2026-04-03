# LLM Formula Reverse-Engineering Benchmark

A benchmark task that evaluates an LLM's ability to reverse-engineer a hidden
mathematical formula from data. The model receives 100 labeled rows with 16
feature columns and must discover the exact scoring function.

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
```

## How It Works

1. **Dataset**: 16 feature columns (x1-x16), only 8 used in the formula
2. **Formula**: 7 terms — trigonometric functions, conditionals, floor, offset
3. **LLM gets**: 100 rows + hints about operation types + 5 conversation turns with code execution
4. **Scoring**: `score = 1 - clamp(MAE / P85_P15, 0, 1)` on 500 hidden rows

## Results

| Model | Score | Approach |
|---|---|---|
| Reference (exact formula) | 1.00 | Ground truth |
| GPT-5.2 | 0.53 | Brute-force numerical fit |
| OLS baseline | 0.56 | Linear regression |

See [SOLUTION.md](SOLUTION.md) for full analysis and human expert walkthrough.

## Testing

```bash
pytest tests/ -v
```

41 tests covering: formula correctness, scoring metric, dataset integrity,
code extraction, reference solution consistency.

## Configuration

Environment variables (`.env`):
- `OPENROUTER_API_KEY` — API key for OpenRouter
- `MODEL_NAME` — model identifier (default: `openai/gpt-5.2`)
- `MAX_CALLS` — number of conversation turns (default: `5`)
