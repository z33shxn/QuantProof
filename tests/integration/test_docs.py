"""Execute the Python examples in the documentation so they cannot go stale."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "docs/api/key-objects.md"]


def _blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


@pytest.mark.parametrize(
    ("path", "index"),
    [(p, i) for p in DOCS for i in range(len(_blocks(p)))],
    ids=lambda v: v.name if isinstance(v, Path) else str(v),
)
def test_doc_example_runs(path: Path, index: int) -> None:
    code = _blocks(path)[index]
    exec(compile(code, f"{path.name}[{index}]", "exec"), {"__name__": "__doc_example__"})
