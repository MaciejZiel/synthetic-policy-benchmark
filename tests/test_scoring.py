"""Tests for the scoring system."""

import math
import sys
import pathlib
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.score_predictions import score_formula, load_hidden_test, FEATURE_COLS
from src.reference_solution import predict as reference_predict
from src.generate_dataset import hidden_formula


class TestReferenceMatchesGenerator:
    """Reference solution must match generate_dataset formula exactly."""

    def test_reference_matches_on_hidden_test(self):
        """Reference predict() must score 1.0 on the hidden test set."""
        hidden = load_hidden_test()
        for _, row in hidden.iterrows():
            kwargs = {col: row[col] for col in FEATURE_COLS}
            ref_score = reference_predict(**kwargs)
            assert ref_score == row["score"], (
                f"Mismatch at id={row['id']}: reference={ref_score}, expected={row['score']}"
            )

    def test_reference_matches_generator_formula(self):
        """reference_solution.predict and generate_dataset.hidden_formula must agree."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            row = pd.Series({f"x{i}": int(rng.integers(1, 50)) for i in range(1, 17)})
            gen_score = hidden_formula(row)
            ref_score = reference_predict(**{f"x{i}": row[f"x{i}"] for i in range(1, 17)})
            assert gen_score == ref_score


class TestScoringMetric:
    """Tests for the score = 1 - clamp(MAE / P85_P15, 0, 1) metric."""

    def test_perfect_score(self):
        """Reference formula should get score=1.0."""
        # Create a temporary model_formula.py with the reference solution
        root = pathlib.Path(__file__).resolve().parent.parent
        tmp_slug = "_test_perfect"
        tmp_dir = root / "artifacts" / "model_runs" / tmp_slug
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            formula_code = (
                "import math\n"
                "def predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16):\n"
                "    return round(\n"
                "        35.0*math.sin(0.5*x3)\n"
                "        +28.0*math.cos(0.4*x9)\n"
                "        +20.0*math.sin(0.3*x6-0.2*x14)\n"
                "        +(5.0 if x4>25 else -5.0)\n"
                "        +(-4.0 if x7+x16>50 else 4.0)\n"
                "        +2.0*math.floor(x11/7.0)\n"
                "        +50\n"
                "    ,2)\n"
            )
            (tmp_dir / "model_formula.py").write_text(formula_code)

            result = score_formula(tmp_slug)
            assert result["score"] == 1.0
            assert result["mae"] == 0.0
            assert result["rows_scored"] == 500
            assert result["rows_failed"] == 0
        finally:
            shutil.rmtree(tmp_dir)

    def test_constant_prediction_low_score(self):
        """A constant prediction should score poorly."""
        root = pathlib.Path(__file__).resolve().parent.parent
        tmp_slug = "_test_constant"
        tmp_dir = root / "artifacts" / "model_runs" / tmp_slug
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            formula_code = (
                "def predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16):\n"
                "    return 50.0\n"
            )
            (tmp_dir / "model_formula.py").write_text(formula_code)

            result = score_formula(tmp_slug)
            assert result["score"] < 0.7
            assert result["rows_scored"] == 500
        finally:
            shutil.rmtree(tmp_dir)

    def test_garbage_prediction_zero_score(self):
        """Wildly wrong predictions should score 0.0."""
        root = pathlib.Path(__file__).resolve().parent.parent
        tmp_slug = "_test_garbage"
        tmp_dir = root / "artifacts" / "model_runs" / tmp_slug
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            formula_code = (
                "def predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16):\n"
                "    return 99999.0\n"
            )
            (tmp_dir / "model_formula.py").write_text(formula_code)

            result = score_formula(tmp_slug)
            assert result["score"] == 0.0
        finally:
            shutil.rmtree(tmp_dir)

    def test_p85_p15_is_positive(self):
        """P85-P15 spread should be positive for a non-trivial formula."""
        hidden = load_hidden_test()
        true_arr = hidden["score"].values
        p85_p15 = float(np.percentile(true_arr, 85) - np.percentile(true_arr, 15))
        assert p85_p15 > 0

    def test_score_in_zero_one_range(self):
        """Score must always be in [0, 1]."""
        root = pathlib.Path(__file__).resolve().parent.parent
        tmp_slug = "_test_range"
        tmp_dir = root / "artifacts" / "model_runs" / tmp_slug
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            formula_code = (
                "def predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16):\n"
                "    return -999.0\n"
            )
            (tmp_dir / "model_formula.py").write_text(formula_code)

            result = score_formula(tmp_slug)
            assert 0.0 <= result["score"] <= 1.0
        finally:
            shutil.rmtree(tmp_dir)


class TestHiddenTestSet:
    """Tests for the hidden test set integrity."""

    def test_hidden_test_has_500_rows(self):
        hidden = load_hidden_test()
        assert len(hidden) == 500

    def test_hidden_test_has_correct_columns(self):
        hidden = load_hidden_test()
        expected = ["id"] + FEATURE_COLS + ["score"]
        assert list(hidden.columns) == expected

    def test_hidden_test_ids_start_at_10001(self):
        hidden = load_hidden_test()
        assert hidden["id"].iloc[0] == 10001
        assert hidden["id"].iloc[-1] == 10500
