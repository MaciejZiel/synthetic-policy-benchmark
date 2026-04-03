"""
Generate synthetic dataset with a hidden scoring formula.

Outputs:
    dataset/history_signals.csv   - 100 rows the LLM sees (with score)
    dataset/target_signals.csv    - 20 rows without score (LLM predicts)
    artifacts/hidden_test.csv     - 500 rows for final scoring (NEVER shown)
    artifacts/reference_answer.json
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED = 42

FEATURE_COLS = [f"x{i}" for i in range(1, 17)]


def hidden_formula(row: pd.Series) -> float:
    """The secret formula the LLM must reverse-engineer."""
    x3 = row["x3"]
    x4 = row["x4"]
    x6 = row["x6"]
    x7 = row["x7"]
    x9 = row["x9"]
    x11 = row["x11"]
    x14 = row["x14"]
    x16 = row["x16"]

    score = (
        # single-variable trig (discoverable via residual plot + FFT)
        35.0 * math.sin(0.5 * x3)
        + 28.0 * math.cos(0.4 * x9)

        # two-variable trig (harder — requires testing combinations)
        + 20.0 * math.sin(0.3 * x6 - 0.2 * x14)

        # simple threshold
        + (5.0 if x4 > 25 else -5.0)

        # threshold on sum of two variables
        + (-4.0 if x7 + x16 > 50 else 4.0)

        # floor with non-obvious divisor
        + 2.0 * math.floor(x11 / 7.0)

        # offset
        + 50
    )
    return round(score, 2)


def _generate_rows(n: int, rng: np.random.Generator, start_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(start_id, start_id + n),
        "x1": rng.integers(1, 50, size=n),
        "x2": rng.integers(1, 60, size=n),
        "x3": rng.integers(1, 50, size=n),
        "x4": rng.integers(1, 50, size=n),
        "x5": rng.integers(1, 40, size=n),
        "x6": rng.integers(1, 50, size=n),
        "x7": rng.integers(1, 45, size=n),
        "x8": rng.integers(1, 55, size=n),
        "x9": rng.integers(1, 50, size=n),
        "x10": rng.integers(1, 60, size=n),
        "x11": rng.integers(1, 50, size=n),
        "x12": rng.integers(1, 45, size=n),
        "x13": rng.integers(1, 55, size=n),
        "x14": rng.integers(1, 50, size=n),
        "x15": rng.integers(1, 40, size=n),
        "x16": rng.integers(1, 50, size=n),
    })


def generate(
    n_history: int = 100,
    n_target: int = 20,
    n_hidden: int = 500,
    seed: int = SEED,
):
    rng = np.random.default_rng(seed)
    history = _generate_rows(n_history, rng, start_id=1)
    target = _generate_rows(n_target, rng, start_id=n_history + 1)

    rng_hidden = np.random.default_rng(seed + 999)
    hidden = _generate_rows(n_hidden, rng_hidden, start_id=10001)

    for df in [history, target, hidden]:
        df["score"] = df.apply(hidden_formula, axis=1)

    return history, target, hidden


def main():
    history, target, hidden = generate()

    dataset_dir = ROOT / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(dataset_dir / "history_signals.csv", index=False)
    target.drop(columns=["score"]).to_csv(dataset_dir / "target_signals.csv", index=False)

    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    hidden.to_csv(artifacts_dir / "hidden_test.csv", index=False)

    ref = {"scores": dict(zip(
        target["id"].astype(str).tolist(),
        target["score"].tolist(),
    ))}
    (artifacts_dir / "reference_answer.json").write_text(
        json.dumps(ref, indent=2) + "\n"
    )

    print(f"History:     {len(history)} rows")
    print(f"Target:      {len(target)} rows")
    print(f"Hidden test: {len(hidden)} rows")
    print(f"Columns:     {len(FEATURE_COLS)} features (x1-x16)")
    print(f"Score range: {history['score'].min():.2f} - {history['score'].max():.2f}")


if __name__ == "__main__":
    main()
