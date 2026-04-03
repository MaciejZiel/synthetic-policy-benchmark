"""
Score a model's formula against the hidden test set (500 rows).

Usage:
    python src/score_predictions.py <model_slug>

Metric: score = 1 - clamp(MAE / P85_P15, 0, 1)
    P85_P15 = 85th percentile - 15th percentile of true scores.
"""

import importlib.util
import json
import math
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

FEATURE_COLS = [f"x{i}" for i in range(1, 17)]


def load_hidden_test() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts" / "hidden_test.csv")


def load_model_formula(model_slug: str):
    formula_path = ROOT / "artifacts" / "model_runs" / model_slug / "model_formula.py"
    if not formula_path.exists():
        raise FileNotFoundError(f"No formula found at {formula_path}")

    spec = importlib.util.spec_from_file_location("model_formula", formula_path)
    mod = importlib.util.module_from_spec(spec)
    mod.math = math
    mod.__builtins__ = __builtins__
    spec.loader.exec_module(mod)

    if not hasattr(mod, "predict"):
        raise ValueError("model_formula.py does not define a predict() function")
    return mod.predict


def score_formula(model_slug: str) -> dict:
    test_df = load_hidden_test()
    predict_fn = load_model_formula(model_slug)

    true_scores = test_df["score"].tolist()
    pred_scores = []
    errors_list = []

    for _, row in test_df.iterrows():
        try:
            pred = predict_fn(**{col: row[col] for col in FEATURE_COLS})
            pred_scores.append(float(pred))
        except Exception as exc:
            errors_list.append(str(exc))
            pred_scores.append(None)

    pairs = [(t, p) for t, p in zip(true_scores, pred_scores) if p is not None]
    if not pairs:
        return {
            "score": 0.0,
            "mae": None,
            "p85_p15": None,
            "rows_scored": 0,
            "rows_failed": len(true_scores),
            "errors": errors_list[:5],
        }

    mae = sum(abs(t - p) for t, p in pairs) / len(pairs)
    true_arr = np.array(true_scores)
    p85_p15 = float(np.percentile(true_arr, 85) - np.percentile(true_arr, 15))

    if p85_p15 == 0:
        final_score = 1.0
    else:
        final_score = max(0.0, min(1.0, 1.0 - mae / p85_p15))

    return {
        "score": round(final_score, 4),
        "mae": round(mae, 4),
        "p85_p15": round(p85_p15, 2),
        "rows_scored": len(pairs),
        "rows_failed": len(true_scores) - len(pairs),
    }


def main() -> None:
    if len(sys.argv) < 2:
        runs_dir = ROOT / "artifacts" / "model_runs"
        candidates = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("No model runs found.")
            sys.exit(1)
        model_slug = candidates[-1].name
    else:
        model_slug = sys.argv[1]

    print(f"Scoring model: {model_slug}")
    print(f"Test set: 500 hidden rows")
    print(f"{'─'*40}")

    try:
        result = score_formula(model_slug)
    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()
        result = {"score": 0.0, "error": str(exc)}

    print(json.dumps(result, indent=2))

    out_path = ROOT / "artifacts" / "model_runs" / model_slug / "scoring_report.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
