"""
Benchmark runner - sends task to an LLM via OpenRouter, executes code,
feeds back results, repeats exactly MAX_CALLS times.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.getenv("MODEL_NAME", "openai/gpt-4o")
MAX_CALLS = int(os.getenv("MAX_CALLS", "5"))
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_llm(messages: list[dict], retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 8192},
                timeout=300,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content or "(empty)"
        except requests.exceptions.ReadTimeout:
            if attempt < retries:
                print(f"  [timeout, retry {attempt+1}/{retries}]")
            else:
                raise


def extract_code(text: str) -> str | None:
    m = re.search(r"```python\s*\n(.+?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def pip_install(packages: list[str]) -> None:
    """Pre-install packages so the model's code can import them."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        capture_output=True, timeout=120,
    )


def run_code(code: str) -> tuple[str, str, int]:
    # Auto-detect missing imports and install them
    common_libs = {
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "statsmodels": "statsmodels",
        "sympy": "sympy",
        "gplearn": "gplearn",
    }
    for mod, pkg in common_libs.items():
        if mod in code:
            pip_install([pkg])

    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout after 120s", 1


def extract_function(text: str) -> str | None:
    """Extract the LAST def predict(...) from text."""
    best = None
    for block in re.findall(r"```python\s*\n(.+?)```", text, re.DOTALL):
        if "def predict(" in block:
            m = re.search(r"(def predict\(.+?)(?=\ndef |\nclass |\Z)", block, re.DOTALL)
            if m:
                best = m.group(1).rstrip()
    if best:
        return best
    for m in re.finditer(r"(def predict\(.+?)(?=\ndef |\nclass |\Z)", text, re.DOTALL):
        best = m.group(1).rstrip()
    return best


def build_system_prompt() -> str:
    history = (ROOT / "dataset" / "history_signals.csv").read_text()
    task = (ROOT / "task_instructions.txt").read_text()
    return f"{task}\n\n=== dataset/history_signals.csv ===\n{history}\n"


def run() -> None:
    system = build_system_prompt()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            "Analyse the data and reverse-engineer the scoring formula. "
            "You may write Python code to explore the data - wrap it in ```python blocks "
            "and I will execute it and show you the output. "
            "You can use any Python library (numpy, pandas, scipy, sklearn, sympy, etc.). "
            "After analysis, output ONLY the final `def predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16)` function."
        )},
    ]

    print(f"Model:     {MODEL}")
    print(f"Max calls: {MAX_CALLS}")
    print("=" * 60)

    log = []
    start = time.time()

    for i in range(1, MAX_CALLS + 1):
        print(f"\n{'─'*60}")
        print(f"  CALL {i}/{MAX_CALLS}")
        print(f"{'─'*60}")

        reply = call_llm(messages)
        messages.append({"role": "assistant", "content": reply})

        print(f"\n  Assistant ({len(reply)} chars):")
        print(f"  {reply[:600]}{'...' if len(reply) > 600 else ''}")

        code = extract_code(reply)
        if code:
            stdout, stderr, rc = run_code(code)
            print(f"\n  CODE OUTPUT (rc={rc})")
            if stdout:
                print(f"  {stdout[:500]}")
            if stderr:
                print(f"  STDERR: {stderr[:300]}")

            log.append({"call": i, "code": code, "stdout": stdout, "stderr": stderr, "rc": rc})

            if i < MAX_CALLS:
                feedback = f"Execution result (rc={rc}):\n"
                if stdout:
                    feedback += f"STDOUT:\n{stdout}\n"
                if stderr:
                    feedback += f"STDERR:\n{stderr}\n"
                if i == MAX_CALLS - 1:
                    feedback += "\nThis is your LAST chance. Output the final `def predict(x1, ..., x16)` function now."
                else:
                    feedback += "\nContinue your analysis or output the final `def predict(x1, ..., x16)` function."
                messages.append({"role": "user", "content": feedback})
        else:
            log.append({"call": i, "code": None, "text": reply[:200]})
            if i < MAX_CALLS:
                messages.append({"role": "user", "content":
                    "Please provide Python code in a ```python block, or output the final `def predict(x1, ..., x16)` function."
                })

    elapsed = time.time() - start

    all_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    func = extract_function(all_text)

    slug = MODEL.replace("/", "_")
    run_dir = ROOT / "artifacts" / "model_runs" / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    if func:
        func_code = func if "import math" in func else "import math\n" + func
        (run_dir / "model_formula.py").write_text(func_code + "\n")

    (run_dir / "run_metadata.json").write_text(json.dumps({
        "model": MODEL,
        "max_calls": MAX_CALLS,
        "calls_used": len(log),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "formula_found": func is not None,
    }, indent=2) + "\n")
    (run_dir / "run_log.json").write_text(json.dumps(log, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"  RUN REPORT")
    print(f"{'='*60}")
    print(f"  Model:       {MODEL}")
    print(f"  Calls:       {len(log)}/{MAX_CALLS}")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"  Formula:     {'FOUND' if func else 'NOT FOUND'}")
    if func:
        print(f"\n{func[:400]}")
    print(f"\n  Artifacts:   {run_dir}")

    # Auto-score the model's formula
    if func:
        print(f"\n{'─'*60}")
        print(f"  SCORING")
        print(f"{'─'*60}")
        from src.score_predictions import score_formula
        try:
            result = score_formula(slug)
            report_path = run_dir / "scoring_report.json"
            report_path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"  Score:       {result['score']}")
            print(f"  MAE:         {result['mae']}")
            print(f"  P85-P15:     {result['p85_p15']}")
            print(f"  Rows scored: {result['rows_scored']}")
            print(f"  Report:      {report_path}")
        except Exception as exc:
            print(f"  Scoring failed: {exc}")
    else:
        print(f"  No formula found — skipping scoring.")

    print(f"{'='*60}")


if __name__ == "__main__":
    run()
