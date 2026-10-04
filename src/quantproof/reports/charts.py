"""Dependency-free inline SVG charts for the HTML report.

Charts render offline (no JavaScript library or CDN). Colors are referenced via CSS
custom properties defined by the report template so light and dark themes swap in
one place. Every mark carries a ``<title>`` for native hover tooltips, and every
chart's numbers are also shown in an adjacent table in the report.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Sequence
from typing import Any

W, H = 720, 260
PAD_L, PAD_R, PAD_T, PAD_B = 56, 120, 16, 32


def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo] if math.isfinite(lo) else [0.0]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt(v: float) -> str:
    if float(v).is_integer() and abs(v) < 1e6:
        return f"{int(v)}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def line_chart(
    series: Sequence[dict[str, Any]], *, y_label: str = "", chart_id: str = "line"
) -> str:
    """Multi-series line chart. Each series: ``{"name", "x": [iso str], "y": [float], "slot": int}``."""
    pts = [
        (s, [(i, y) for i, y in enumerate(s["y"]) if y is not None and math.isfinite(y)])
        for s in series
    ]
    pts = [(s, p) for s, p in pts if p]
    if not pts:
        return ""
    n = max(len(s["x"]) for s, _ in pts)
    ys = [y for _, p in pts for _, y in p]
    lo, hi = min(ys), max(ys)
    # Growth curves spanning more than ~1.5 orders of magnitude are drawn on a log scale.
    log = lo > 0 and hi / lo > 30
    if log:
        lo, hi = math.log10(lo), math.log10(hi)
        ticks = [float(t) for t in range(math.floor(lo), math.ceil(hi) + 1)]
    else:
        if hi == lo:
            hi, lo = hi + 1, lo - 1
        ticks = _nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def tr(v: float) -> float:
        return math.log10(v) if log else v

    def sx(i: int) -> float:
        return PAD_L + pw * i / max(n - 1, 1)

    def sy(v: float) -> float:
        return PAD_T + ph * (1 - (tr(v) - lo) / (hi - lo))

    out = [
        f'<svg class="qp-chart qp-line" id="{_e(chart_id)}" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="{_e(y_label)} over time">'
    ]
    for t in ticks:
        value = 10**t if log else t
        y = sy(value)
        out.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(
            f'<text class="tick" x="{PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end">{_fmt(value)}</text>'
        )
    if log:
        out.append(f'<text class="tick" x="{PAD_L + 4}" y="{PAD_T + 10}">log scale</text>')
    xs = pts[0][0]["x"]
    for frac in (0, 0.5, 1):
        i = round(frac * (len(xs) - 1))
        anchor = {0: "start", 0.5: "middle", 1: "end"}[frac]
        out.append(
            f'<text class="tick" x="{sx(i):.1f}" y="{H - 10}" text-anchor="{anchor}">{_e(str(xs[i])[:10])}</text>'
        )
    label_y: list[float] = []
    for s, p in pts:
        d = " ".join(
            f"{'M' if k == 0 else 'L'}{sx(i):.1f},{sy(v):.1f}" for k, (i, v) in enumerate(p)
        )
        slot = s.get("slot", 1)
        out.append(f'<path class="series s{slot}" d="{d}"><title>{_e(s["name"])}</title></path>')
        last_i, last_v = p[-1]
        y = sy(last_v)
        for prev in label_y:
            if abs(prev - y) < 13:
                y = prev + 13 if y >= prev else prev - 13
        label_y.append(y)
        out.append(
            f'<text class="direct-label" x="{sx(last_i) + 6:.1f}" y="{y + 4:.1f}">'
            f"{_e(s.get('short', s['name']))} {_fmt(last_v)}</text>"
        )
    out.append(
        f'<line class="crosshair" x1="0" x2="0" y1="{PAD_T}" y2="{H - PAD_B}" visibility="hidden"/>'
        f'<rect class="hit" x="{PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}" fill="transparent"/>'
    )
    out.append("</svg>")
    data = {
        "x": xs,
        "series": [{"name": s["name"], "y": s["y"], "slot": s.get("slot", 1)} for s, _ in pts],
        "geom": {"left": PAD_L, "width": pw, "W": W},
    }
    payload = json.dumps(data).replace("</", "<\\/")
    out.append(
        f'<script type="application/json" class="qp-line-data" data-for="{_e(chart_id)}">'
        f"{payload}</script>"
    )
    return "".join(out)


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    y_label: str = "",
    highlight: int | None = None,
) -> str:
    """Vertical bars from a zero baseline (supports negative values)."""
    vals = [v if v is not None and math.isfinite(v) else None for v in values]
    finite = [v for v in vals if v is not None]
    if not finite:
        return ""
    lo, hi = min(0.0, *finite), max(0.0, *finite)
    if hi == lo:
        hi = lo + 1
    ticks = _nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    w, h, pl, pr = 520, 220, 48, 12
    pw, ph = w - pl - pr, h - PAD_T - PAD_B

    def sy(v: float) -> float:
        return PAD_T + ph * (1 - (v - lo) / (hi - lo))

    n = len(vals)
    slot_w = pw / n
    bw = max(6.0, min(48.0, slot_w - 8))
    out = [
        f'<svg class="qp-chart" viewBox="0 0 {w} {h}" style="max-width:{w}px" role="img" aria-label="{_e(y_label)}">'
    ]
    for t in ticks:
        out.append(
            f'<line class="grid" x1="{pl}" x2="{w - pr}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{pl - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>'
        )
    zero = sy(0.0)
    for i, (lab, v) in enumerate(zip(labels, vals, strict=True)):
        cx = pl + slot_w * (i + 0.5)
        out.append(
            f'<text class="tick" x="{cx:.1f}" y="{h - 10}" text-anchor="middle">{_e(lab)}</text>'
        )
        if v is None:
            continue
        top, bottom = (sy(v), zero) if v >= 0 else (zero, sy(v))
        cls = "bar s1" + (" hl" if highlight == i else "") + (" neg" if v < 0 else "")
        hgt = max(bottom - top, 1.0)
        r = min(4.0, hgt / 2, bw / 2)
        out.append(
            f'<rect class="{cls}" x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{hgt:.1f}" '
            f'rx="{r:.1f}"><title>{_e(lab)}: {v:.3f}</title></rect>'
        )
    out.append(f'<line class="baseline" x1="{pl}" x2="{w - pr}" y1="{zero:.1f}" y2="{zero:.1f}"/>')
    out.append("</svg>")
    return "".join(out)


def histogram(
    values: Sequence[float], *, bins: int = 20, vline: float | None = 0.0, label: str = ""
) -> str:
    """Histogram of finite values with an optional reference line."""
    v = [x for x in values if x is not None and math.isfinite(x)]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    if vline is not None:
        lo, hi = min(lo, vline), max(hi, vline)
    if hi == lo:
        hi = lo + 1
    width = (hi - lo) / bins
    counts = [0] * bins
    for x in v:
        counts[min(int((x - lo) / width), bins - 1)] += 1
    w, h, pl, pr = 520, 220, 40, 12
    pw, ph = w - pl - pr, h - PAD_T - PAD_B
    top = max(counts)
    out = [
        f'<svg class="qp-chart" viewBox="0 0 {w} {h}" style="max-width:{w}px" role="img" aria-label="{_e(label)}">'
    ]
    for t in _nice_ticks(0, top, 4):
        y = PAD_T + ph * (1 - t / top) if top else PAD_T
        if t <= top:
            out.append(f'<line class="grid" x1="{pl}" x2="{w - pr}" y1="{y:.1f}" y2="{y:.1f}"/>')
            out.append(
                f'<text class="tick" x="{pl - 6}" y="{y + 4:.1f}" text-anchor="end">{t:.0f}</text>'
            )
    bw = pw / bins
    for i, c in enumerate(counts):
        if not c:
            continue
        hgt = ph * c / top
        x0 = lo + i * width
        out.append(
            f'<rect class="bar s1" x="{pl + i * bw + 1:.1f}" y="{PAD_T + ph - hgt:.1f}" width="{bw - 2:.1f}" '
            f'height="{hgt:.1f}" rx="{min(4.0, (bw - 2) / 2):.1f}"><title>[{x0:.2f}, {x0 + width:.2f}): {c}</title></rect>'
        )
    for frac in (0, 1):
        x = pl + pw * frac
        out.append(
            f'<text class="tick" x="{x:.1f}" y="{h - 10}" text-anchor="{"start" if frac == 0 else "end"}">'
            f"{_fmt(lo + (hi - lo) * frac)}</text>"
        )
    if vline is not None:
        x = pl + pw * (vline - lo) / (hi - lo)
        out.append(
            f'<line class="refline" x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{PAD_T + ph}"/>'
        )
        out.append(f'<text class="tick" x="{x + 4:.1f}" y="{PAD_T + 10}">{_fmt(vline)}</text>')
    out.append(
        f'<line class="baseline" x1="{pl}" x2="{w - pr}" y1="{PAD_T + ph}" y2="{PAD_T + ph}"/>'
    )
    out.append("</svg>")
    return "".join(out)


def _diverging(v: float, vmax: float) -> str:
    """Blue (positive) ↔ gray ↔ red (negative), 5 steps per arm."""
    if not math.isfinite(v) or vmax <= 0:
        return "var(--div-mid)"
    t = max(-1.0, min(1.0, v / vmax))
    step = min(5, int(abs(t) * 5 + 0.5))
    if step == 0:
        return "var(--div-mid)"
    return f"var(--div-{'pos' if t > 0 else 'neg'}-{step})"


def heatmap(
    x_values: Sequence[Any],
    y_values: Sequence[Any],
    values: Sequence[Sequence[float | None]],
    *,
    x_label: str,
    y_label: str,
) -> str:
    """Parameter-surface heat-map with a diverging scale centred on zero; ★ marks the maximum."""
    flat = [v for row in values for v in row if v is not None and math.isfinite(v)]
    if not flat:
        return ""
    vmax = max(abs(min(flat)), abs(max(flat)))
    best = max(flat)
    cw, chh, pl, pt = 64, 30, 76, 28
    w = pl + cw * len(x_values) + 8
    h = pt + chh * len(y_values) + 34
    out = [
        f'<svg class="qp-chart qp-heat" viewBox="0 0 {w} {h}" style="max-width:{w}px" role="img" '
        f'aria-label="Sharpe by {_e(x_label)} and {_e(y_label)}">'
    ]
    for j, xv in enumerate(x_values):
        out.append(
            f'<text class="tick" x="{pl + cw * (j + 0.5):.1f}" y="{pt - 8}" text-anchor="middle">{_e(_fmt(float(xv)))}</text>'
        )
    for i, yv in enumerate(y_values):
        out.append(
            f'<text class="tick" x="{pl - 8}" y="{pt + chh * (i + 0.5) + 4:.1f}" text-anchor="end">{_e(_fmt(float(yv)))}</text>'
        )
        for j, _ in enumerate(x_values):
            v = values[i][j]
            x, y = pl + cw * j, pt + chh * i
            if v is None or not math.isfinite(v):
                out.append(
                    f'<rect class="cell empty" x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{chh - 2}" rx="3"/>'
                )
                continue
            star = "★" if v == best else ""
            out.append(
                f'<rect class="cell" x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{chh - 2}" rx="3" '
                f'style="fill:{_diverging(v, vmax)}"><title>{_e(x_label)}={_e(x_values[j])}, '
                f"{_e(y_label)}={_e(y_values[i])}: Sharpe {v:.2f}</title></rect>"
            )
            cls = "cell-label strong" if abs(v) / vmax > 0.55 else "cell-label"
            out.append(
                f'<text class="{cls}" x="{x + cw / 2:.1f}" y="{y + chh / 2 + 4:.1f}" text-anchor="middle">{v:.2f}{star}</text>'
            )
    out.append(
        f'<text class="axis-title" x="{pl + cw * len(x_values) / 2:.1f}" y="{h - 8}" text-anchor="middle">{_e(x_label)}</text>'
    )
    mid_y = pt + chh * len(y_values) / 2
    out.append(
        f'<text class="axis-title" x="14" y="{mid_y:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {mid_y:.1f})">{_e(y_label)}</text>'
    )
    out.append("</svg>")
    return "".join(out)
