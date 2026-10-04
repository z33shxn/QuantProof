"""The example notebook's code cells must run top to bottom."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_walkthrough_notebook_executes(monkeypatch):
    nb = json.loads((ROOT / "examples/notebooks/walkthrough.ipynb").read_text())
    monkeypatch.chdir(ROOT)
    namespace: dict[str, object] = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "walkthrough.ipynb", "exec"), namespace)
    assert namespace["result"].status.value == "FAIL"  # type: ignore[attr-defined]
    assert os.getcwd() == str(ROOT)
