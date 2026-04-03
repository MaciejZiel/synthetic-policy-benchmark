"""Tests for benchmark runner utilities (no API calls)."""

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.run_benchmark import extract_code, extract_function


class TestExtractCode:
    """Tests for extracting Python code blocks from LLM responses."""

    def test_basic_code_block(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        assert extract_code(text) == "print('hello')"

    def test_multiline_code_block(self):
        text = "```python\nimport math\nx = math.sin(1)\nprint(x)\n```"
        assert extract_code(text) == "import math\nx = math.sin(1)\nprint(x)"

    def test_no_code_block(self):
        text = "I think the answer is 42."
        assert extract_code(text) is None

    def test_non_python_block_ignored(self):
        text = "```json\n{\"a\": 1}\n```"
        assert extract_code(text) is None

    def test_first_block_returned(self):
        text = "```python\nfirst()\n```\n\n```python\nsecond()\n```"
        assert extract_code(text) == "first()"


class TestExtractFunction:
    """Tests for extracting the predict() function from LLM output."""

    def test_basic_function(self):
        text = '```python\ndef predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):\n    return 42\n```'
        func = extract_function(text)
        assert func is not None
        assert "def predict(" in func
        assert "return 42" in func

    def test_no_function(self):
        text = "```python\nprint('hello')\n```"
        assert extract_function(text) is None

    def test_last_function_wins(self):
        text = (
            "```python\ndef predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):\n    return 1\n```\n\n"
            "```python\ndef predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):\n    return 2\n```"
        )
        func = extract_function(text)
        assert "return 2" in func

    def test_function_with_math(self):
        text = '```python\nimport math\ndef predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):\n    return math.sin(x1) + 50\n```'
        func = extract_function(text)
        assert func is not None
        assert "math.sin(x1)" in func

    def test_function_outside_code_block(self):
        text = "def predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):\n    return 0"
        func = extract_function(text)
        assert func is not None


class TestDatasetFiles:
    """Tests that generated dataset files exist and are well-formed."""

    ROOT = pathlib.Path(__file__).resolve().parent.parent

    def test_history_signals_exists(self):
        path = self.ROOT / "dataset" / "history_signals.csv"
        assert path.exists()

    def test_target_signals_exists(self):
        path = self.ROOT / "dataset" / "target_signals.csv"
        assert path.exists()

    def test_target_signals_has_no_score(self):
        """Target CSV must not contain the score column."""
        import pandas as pd
        path = self.ROOT / "dataset" / "target_signals.csv"
        df = pd.read_csv(path)
        assert "score" not in df.columns

    def test_hidden_test_exists(self):
        path = self.ROOT / "artifacts" / "hidden_test.csv"
        assert path.exists()

    def test_reference_answer_exists(self):
        path = self.ROOT / "artifacts" / "reference_answer.json"
        assert path.exists()

    def test_reference_answer_valid_json(self):
        import json
        path = self.ROOT / "artifacts" / "reference_answer.json"
        data = json.loads(path.read_text())
        assert "scores" in data
        assert len(data["scores"]) == 20  # 20 target rows
