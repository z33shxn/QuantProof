"""Report renderers: HTML, Markdown, JSON, and plain text."""

from pathlib import Path
from typing import TYPE_CHECKING

from quantproof.reports.html import render_html
from quantproof.reports.json import load_result, render_json, result_to_dict
from quantproof.reports.markdown import render_markdown
from quantproof.reports.text import render_text

if TYPE_CHECKING:
    from quantproof.results import AuditResult

FORMATS = ("html", "json", "markdown", "text")


def render(result: "AuditResult", fmt: str) -> str:
    """Render ``result`` in one of :data:`FORMATS`."""
    if fmt == "html":
        return render_html(result)
    if fmt == "json":
        return render_json(result)
    if fmt in ("markdown", "md"):
        return render_markdown(result)
    if fmt == "text":
        return render_text(result)
    raise ValueError(f"Unknown report format {fmt!r}; choose from {FORMATS}.")


def write_report(result: "AuditResult", path: str | Path, fmt: str | None = None) -> Path:
    """Write a report; the format is inferred from the extension when not given."""
    p = Path(path)
    if fmt is None:
        fmt = {
            ".html": "html",
            ".htm": "html",
            ".json": "json",
            ".md": "markdown",
            ".txt": "text",
        }.get(p.suffix.lower(), "html")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(result, fmt), encoding="utf-8")
    return p


__all__ = [
    "FORMATS",
    "load_result",
    "render",
    "render_html",
    "render_json",
    "render_markdown",
    "render_text",
    "result_to_dict",
    "write_report",
]
