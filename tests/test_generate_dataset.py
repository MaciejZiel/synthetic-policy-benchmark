"""Tests for dataset generation."""

import math
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.generate_dataset import hidden_formula, generate, FEATURE_COLS


class TestHiddenFormula:
    """Tests for the hidden scoring formula."""

    def test_deterministic(self):
        """Same input always produces the same output."""
        row = pd.Series({f"x{i}": 25 for i in range(1, 17)})
        assert hidden_formula(row) == hidden_formula(row)

    def test_returns_float(self):
        row = pd.Series({f"x{i}": 10 for i in range(1, 17)})
        result = hidden_formula(row)
        assert isinstance(result, float)

    def test_rounded_to_2_decimals(self):
        row = pd.Series({f"x{i}": 17 for i in range(1, 17)})
        result = hidden_formula(row)
        assert result == round(result, 2)

    def test_known_value(self):
        """Verify formula against a hand-computed value."""
        row = pd.Series({f"x{i}": 0 for i in range(1, 17)})
        row["x3"] = 10
        row["x4"] = 30
        row["x6"] = 20
        row["x7"] = 30
        row["x9"] = 15
        row["x11"] = 21
        row["x14"] = 25
        row["x16"] = 25

        expected = (
            35.0 * math.sin(0.5 * 10)
            + 28.0 * math.cos(0.4 * 15)
            + 20.0 * math.sin(0.3 * 20 - 0.2 * 25)
            + 5.0        # x4=30 > 25
            + (-4.0)     # x7+x16=55 > 50
            + 2.0 * math.floor(21 / 7.0)
            + 50
        )
        assert hidden_formula(row) == round(expected, 2)

    def test_threshold_x4_below(self):
        """x4 <= 25 gives -5."""
        row = pd.Series({f"x{i}": 25 for i in range(1, 17)})
        row["x4"] = 20
        s1 = hidden_formula(row)
        row["x4"] = 30
        s2 = hidden_formula(row)
        assert s2 - s1 == pytest.approx(10.0, abs=0.01)  # +5 - (-5) = 10

    def test_threshold_x7_x16_above(self):
        """x7 + x16 > 50 gives -4, otherwise +4."""
        row = pd.Series({f"x{i}": 1 for i in range(1, 17)})
        row["x7"] = 10
        row["x16"] = 10  # sum=20 <= 50 → +4
        s1 = hidden_formula(row)

        row["x7"] = 30
        row["x16"] = 30  # sum=60 > 50 → -4
        s2 = hidden_formula(row)
        assert s1 - s2 == pytest.approx(8.0, abs=0.01)  # +4 - (-4) = 8

    def test_floor_term(self):
        """floor(x11 / 7) produces correct step values."""
        row = pd.Series({f"x{i}": 1 for i in range(1, 17)})
        row["x11"] = 6
        s1 = hidden_formula(row)
        row["x11"] = 7
        s2 = hidden_formula(row)
        assert s2 - s1 == pytest.approx(2.0, abs=0.01)  # floor jumps by 1 → +2

    def test_unused_columns_dont_affect_score(self):
        """Changing x1, x2, x5, x8, x10, x12, x13, x15 should not change score."""
        row = pd.Series({f"x{i}": 20 for i in range(1, 17)})
        base_score = hidden_formula(row)

        unused = [1, 2, 5, 8, 10, 12, 13, 15]
        for col_idx in unused:
            modified = row.copy()
            modified[f"x{col_idx}"] = 45
            assert hidden_formula(modified) == base_score, f"x{col_idx} should not affect score"


class TestGenerate:
    """Tests for the generate() function."""

    def test_row_counts(self):
        history, target, hidden = generate()
        assert len(history) == 100
        assert len(target) == 20
        assert len(hidden) == 500

    def test_columns(self):
        history, target, hidden = generate()
        expected_cols = ["id"] + FEATURE_COLS + ["score"]
        assert list(history.columns) == expected_cols
        assert list(hidden.columns) == expected_cols

    def test_target_has_no_score_leak(self):
        """Target set should have score computed but generate() caller drops it."""
        _, target, _ = generate()
        assert "score" in target.columns  # generate() returns it with score
        # But main() drops it before saving — tested in test_dataset_files

    def test_feature_ranges(self):
        history, _, _ = generate()
        for col in FEATURE_COLS:
            assert history[col].min() >= 1
            assert history[col].max() <= 60  # max range is 60

    def test_reproducible(self):
        h1, _, _ = generate(seed=42)
        h2, _, _ = generate(seed=42)
        pd.testing.assert_frame_equal(h1, h2)

    def test_different_seeds_differ(self):
        h1, _, _ = generate(seed=42)
        h2, _, _ = generate(seed=99)
        assert not h1["score"].equals(h2["score"])

    def test_ids_are_sequential(self):
        history, target, hidden = generate()
        assert list(history["id"]) == list(range(1, 101))
        assert list(target["id"]) == list(range(101, 121))
        assert list(hidden["id"]) == list(range(10001, 10501))
