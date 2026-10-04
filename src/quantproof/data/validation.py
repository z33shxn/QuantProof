"""Data-quality checks for price data (rule ids ``QP-DATA-001`` … ``QP-DATA-013``).

Every check emits either a ``PASS`` finding or an issue finding with counts,
the affected range, why it matters, and how to fix it. Thresholds come from
:class:`quantproof.config.DataValidationConfig`; individual rules can be
disabled because some datasets legitimately contain, e.g., irregular sampling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantproof._utils import datetime_ns
from quantproof.config import DataValidationConfig
from quantproof.data.loaders import ParsedTimestamps, parse_timestamps, resolve_timestamp_column
from quantproof.data.panel import find_symbol_column, flatten_multiindex
from quantproof.data.schemas import (
    OHLC_COLUMNS,
    PRICE_COLUMNS,
    normalize_column_name,
    primary_price_column,
)
from quantproof.results import Category, Finding, Location
from quantproof.severity import Confidence, Severity


@dataclass
class _Context:
    frame: pd.DataFrame
    ts: pd.DatetimeIndex
    parsed: ParsedTimestamps
    config: DataValidationConfig


def _fmt(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat() if ts is not None and not pd.isna(ts) else "?"


def _range(ts: pd.DatetimeIndex) -> Location | None:
    valid = ts[~np.asarray(ts.isna())]
    if len(valid) == 0:
        return None
    return Location(start=_fmt(valid.min()), end=_fmt(valid.max()))


def _pass(rule_id: str, title: str, message: str) -> Finding:
    return Finding(
        id=rule_id, category=Category.DATA, severity=Severity.PASS, title=title, message=message
    )


def _check_invalid_timestamps(ctx: _Context) -> list[Finding]:
    n = ctx.parsed.n_invalid
    if n == 0:
        return [_pass("QP-DATA-001", "Timestamps parse", "All timestamps parsed successfully.")]
    return [
        Finding(
            id="QP-DATA-001",
            category=Category.DATA,
            severity=Severity.WARN,
            title="Invalid timestamps",
            message=f"{n} timestamp(s) could not be parsed and the rows were dropped.",
            evidence={"count": n, "examples": ctx.parsed.invalid_examples},
            why_it_matters=(
                "Rows without a valid timestamp cannot be placed in time; keeping them would "
                "misalign returns and signals."
            ),
            recommendation="Fix or remove malformed timestamps at the source.",
            confidence=Confidence.HIGH,
        )
    ]


def _check_monotonic(ctx: _Context) -> list[Finding]:
    ts = ctx.ts[~np.asarray(ctx.ts.isna())]
    if len(ts) < 2:
        return []
    values = datetime_ns(ts)
    back = np.flatnonzero(np.diff(values) < 0)
    if back.size == 0:
        return [
            _pass(
                "QP-DATA-002", "Timestamps ordered", "Timestamps are monotonically non-decreasing."
            )
        ]
    first = int(back[0])
    return [
        Finding(
            id="QP-DATA-002",
            category=Category.DATA,
            severity=Severity.WARN,
            title="Non-monotonic timestamps",
            message=(
                f"Expected monotonically increasing timestamps, but found {back.size} out-of-order "
                f"observation(s). First problematic timestamp: {_fmt(ts[first + 1])}; previous "
                f"timestamp: {_fmt(ts[first])}. Rows were sorted before analysis."
            ),
            evidence={
                "count": int(back.size),
                "first_problem": _fmt(ts[first + 1]),
                "previous": _fmt(ts[first]),
            },
            location=Location(start=_fmt(ts[first]), end=_fmt(ts[first + 1])),
            why_it_matters=(
                "Out-of-order rows make shift/rolling operations reference the wrong neighbours, "
                "which can silently introduce look-ahead."
            ),
            recommendation="Sort by timestamp at ingestion and investigate why ordering broke.",
            confidence=Confidence.HIGH,
        )
    ]


def _check_required_ohlc(ctx: _Context) -> list[Finding]:
    present = [c for c in OHLC_COLUMNS if c in ctx.frame.columns]
    missing = [c for c in OHLC_COLUMNS if c not in ctx.frame.columns]
    price_col = primary_price_column(ctx.frame.columns)
    if price_col is None:
        return [
            Finding(
                id="QP-DATA-003",
                category=Category.DATA,
                severity=Severity.FAIL,
                title="No price column",
                message="No 'close', 'adj_close' or 'price' column found.",
                evidence={"columns": list(map(str, ctx.frame.columns))},
                why_it_matters="Returns cannot be computed without a price column.",
                recommendation="Rename the price column to 'close'.",
                confidence=Confidence.HIGH,
            )
        ]
    if not missing:
        return [_pass("QP-DATA-003", "OHLC fields present", "open/high/low/close are all present.")]
    severity = Severity.FAIL if ctx.config.require_ohlc else Severity.INFO
    return [
        Finding(
            id="QP-DATA-003",
            category=Category.DATA,
            severity=severity,
            title="Missing OHLC fields",
            message=f"Missing column(s): {', '.join(missing)}. OHLC consistency checks are limited.",
            evidence={"missing": missing, "present": present, "price_column": price_col},
            why_it_matters=(
                "Without open/high/low, next-open fills and intrabar consistency cannot be checked."
            ),
            recommendation="Provide full OHLC bars if the strategy relies on intrabar prices.",
            confidence=Confidence.HIGH,
        )
    ]


def _check_duplicate_timestamps(ctx: _Context) -> list[Finding]:
    ts = ctx.ts[~np.asarray(ctx.ts.isna())]
    dup_mask = ts.duplicated(keep=False)
    n_dup = int(ts.duplicated(keep="first").sum())
    if n_dup == 0:
        return [_pass("QP-DATA-004", "Unique timestamps", "No duplicated timestamps found.")]
    affected = ts[dup_mask]
    severity = Severity.INFO if ctx.config.allow_duplicate_timestamps else Severity.WARN
    return [
        Finding(
            id="QP-DATA-004",
            category=Category.DATA,
            severity=severity,
            title="Duplicate timestamps",
            message=(
                f"Price series contains {n_dup} duplicated timestamp(s). "
                + (
                    "Duplicates are allowed by configuration."
                    if ctx.config.allow_duplicate_timestamps
                    else "The last observation per timestamp was kept for analysis."
                )
            ),
            evidence={"count": n_dup, "examples": [_fmt(t) for t in affected.unique()[:5]]},
            location=Location(start=_fmt(affected.min()), end=_fmt(affected.max())),
            why_it_matters=(
                "Duplicate observations can distort returns, rolling features, signal timing, "
                "and turnover calculations."
            ),
            recommendation=(
                "De-duplicate at the source, or set data.allow_duplicate_timestamps=true if the "
                "dataset legitimately contains several records per timestamp."
            ),
            confidence=Confidence.HIGH,
        )
    ]


def _check_duplicate_rows(ctx: _Context) -> list[Finding]:
    tmp = ctx.frame.copy()
    tmp["__ts__"] = ctx.ts
    n = int(tmp.duplicated(keep="first").sum())
    if n == 0:
        return [_pass("QP-DATA-005", "No duplicated rows", "No fully duplicated observations.")]
    return [
        Finding(
            id="QP-DATA-005",
            category=Category.DATA,
            severity=Severity.WARN,
            title="Duplicated observations",
            message=f"{n} row(s) are exact duplicates (same timestamp and values).",
            evidence={"count": n},
            why_it_matters="Exact duplicates usually indicate a faulty merge or double ingestion.",
            recommendation="Remove duplicated rows and audit the ingestion pipeline.",
            confidence=Confidence.HIGH,
        )
    ]


def _check_ohlc_relationships(ctx: _Context) -> list[Finding]:
    if not all(c in ctx.frame.columns for c in OHLC_COLUMNS):
        return []
    f = ctx.frame[list(OHLC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    complete = f.notna().all(axis=1)
    o, h, lo, c = (f[k] for k in OHLC_COLUMNS)
    eps = 1e-12
    bad = complete & ((h + eps < np.maximum(o, c)) | (lo - eps > np.minimum(o, c)) | (h + eps < lo))
    n = int(bad.sum())
    if n == 0:
        return [
            _pass(
                "QP-DATA-006",
                "OHLC relationships valid",
                "low ≤ min(open, close) and high ≥ max(open, close) for every complete bar.",
            )
        ]
    bad_ts = ctx.ts[bad.to_numpy()]
    return [
        Finding(
            id="QP-DATA-006",
            category=Category.DATA,
            severity=Severity.FAIL,
            title="Impossible OHLC relationships",
            message=f"{n} bar(s) violate low ≤ open/close ≤ high.",
            evidence={"count": n, "examples": [_fmt(t) for t in bad_ts[:5]]},
            location=_range(bad_ts),
            why_it_matters=(
                "Impossible bars indicate corrupted data; any fill or stop logic using them is "
                "unreliable."
            ),
            recommendation="Repair or drop the affected bars at the source.",
            confidence=Confidence.HIGH,
        )
    ]


def _price_cols(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in PRICE_COLUMNS if c in frame.columns]
    if not cols and "price" in frame.columns:
        cols = ["price"]
    return cols


def _check_nonpositive(ctx: _Context) -> list[Finding]:
    cols = _price_cols(ctx.frame)
    if not cols:
        return []
    values = ctx.frame[cols].apply(pd.to_numeric, errors="coerce")
    bad = (values <= 0).any(axis=1)
    n = int(bad.sum())
    if n == 0:
        return [_pass("QP-DATA-007", "Prices positive", "All prices are strictly positive.")]
    bad_ts = ctx.ts[bad.to_numpy()]
    return [
        Finding(
            id="QP-DATA-007",
            category=Category.DATA,
            severity=Severity.FAIL,
            title="Non-positive prices",
            message=f"{n} row(s) contain zero or negative prices in {', '.join(cols)}.",
            evidence={"count": n, "columns": cols},
            location=_range(bad_ts),
            why_it_matters="Returns from non-positive prices are undefined or infinite.",
            recommendation=(
                "Fix the data. If the instrument can trade at negative prices (e.g. some spreads), "
                "model P&L in price differences instead of percentage returns."
            ),
            confidence=Confidence.HIGH,
        )
    ]


def _check_nulls(ctx: _Context) -> list[Finding]:
    counts = ctx.frame.isna().sum()
    counts = counts[counts > 0]
    if counts.empty:
        return [_pass("QP-DATA-008", "No missing values", "No missing values in any column.")]
    return [
        Finding(
            id="QP-DATA-008",
            category=Category.DATA,
            severity=Severity.WARN,
            title="Unexpected missing values",
            message="Missing values: " + ", ".join(f"{k}={int(v)}" for k, v in counts.items()),
            evidence={"by_column": {str(k): int(v) for k, v in counts.items()}},
            why_it_matters=(
                "Missing values propagate through returns and rolling features; implicit filling "
                "choices (especially backward fill) can leak future information."
            ),
            recommendation="Decide explicitly how to treat gaps; never back-fill prices.",
            confidence=Confidence.HIGH,
        )
    ]


def _check_gaps(ctx: _Context) -> list[Finding]:
    ts = ctx.ts[~np.asarray(ctx.ts.isna())].sort_values().unique()
    if len(ts) < 3:
        return []
    deltas = np.diff(datetime_ns(ts))
    median = float(np.median(deltas))
    if median <= 0:
        return []
    day_ns = 86_400 * 1e9
    threshold = ctx.config.max_gap_multiple * median
    large = deltas > threshold
    session_breaks = 0
    if median < day_ns:
        # Intraday data: gaps that cross a calendar day are treated as session breaks.
        days = datetime_ns(ts.normalize())
        crosses_day = np.diff(days) > 0
        session_breaks = int((large & crosses_day).sum())
        large = large & ~crosses_day
    n = int(large.sum())
    evidence = {
        "median_spacing": str(pd.Timedelta(median, unit="ns")),
        "threshold": str(pd.Timedelta(threshold, unit="ns")),
        "count": n,
        "session_breaks_ignored": session_breaks,
    }
    if n == 0:
        return [
            Finding(
                id="QP-DATA-009",
                category=Category.DATA,
                severity=Severity.PASS,
                title="No unexplained gaps",
                message=(
                    f"No gaps longer than {ctx.config.max_gap_multiple:g}× the median spacing "
                    f"({evidence['median_spacing']})."
                ),
                evidence=evidence,
            )
        ]
    idx = np.flatnonzero(large)
    biggest = int(idx[np.argmax(deltas[idx])])
    evidence["largest_gap"] = str(pd.Timedelta(int(deltas[biggest]), unit="ns"))
    evidence["largest_gap_start"] = _fmt(ts[biggest])
    return [
        Finding(
            id="QP-DATA-009",
            category=Category.DATA,
            severity=Severity.WARN,
            title="Unexplained gaps",
            message=(
                f"{n} gap(s) exceed {ctx.config.max_gap_multiple:g}× the median spacing; the largest "
                f"is {evidence['largest_gap']} starting {evidence['largest_gap_start']}."
            ),
            evidence=evidence,
            location=Location(start=_fmt(ts[idx[0]]), end=_fmt(ts[idx[-1] + 1])),
            why_it_matters=(
                "Gaps make one 'bar' span a long period: returns, volatility estimates, and "
                "bar-count based lags no longer mean what the strategy assumes."
            ),
            recommendation=(
                "Confirm the gaps are genuine market closures; otherwise back-fill the source data "
                "(not the prices) or raise data.max_gap_multiple."
            ),
            confidence=Confidence.MEDIUM,
        )
    ]


def _check_timezone(ctx: _Context) -> list[Finding]:
    p = ctx.parsed
    if p.n_naive and p.n_aware:
        return [
            Finding(
                id="QP-DATA-010",
                category=Category.DATA,
                severity=Severity.WARN,
                title="Mixed timezone-aware and naive timestamps",
                message=(
                    f"{p.n_aware} timestamp(s) carry a UTC offset and {p.n_naive} do not. Naive "
                    "values were interpreted as UTC."
                ),
                evidence={"n_aware": p.n_aware, "n_naive": p.n_naive, "offsets": p.offsets},
                why_it_matters=(
                    "Mixing conventions can shift some bars by hours, misordering events relative "
                    "to signals."
                ),
                recommendation="Store all timestamps in UTC with an explicit offset.",
                confidence=Confidence.HIGH,
            )
        ]
    if len(p.offsets) > 1 and p.n_aware:
        return [
            Finding(
                id="QP-DATA-010",
                category=Category.DATA,
                severity=Severity.INFO,
                title="Multiple UTC offsets",
                message=(
                    f"Timestamps use {len(p.offsets)} distinct UTC offsets ({', '.join(p.offsets)}). "
                    "This is expected across daylight-saving changes; values were normalized to UTC."
                ),
                evidence={"offsets": p.offsets},
            )
        ]
    if p.tz_aware:
        msg = "Timestamps are timezone-aware and consistent."
    else:
        msg = "Timestamps are timezone-naive; they are assumed to be in one consistent local time."
    return [_pass("QP-DATA-010", "Timezone handling consistent", msg)]


def _check_stale(ctx: _Context) -> list[Finding]:
    col = primary_price_column(ctx.frame.columns)
    if col is None:
        return []
    order = np.argsort(datetime_ns(ctx.ts), kind="mergesort")
    s = pd.to_numeric(ctx.frame[col], errors="coerce").to_numpy()[order]
    if s.size < 2:
        return []
    same = np.r_[False, (s[1:] == s[:-1]) & ~np.isnan(s[1:])]
    # Run-length of consecutive repeated values: a run of k identical prices has k-1 `same` flags.
    run_id = np.cumsum(~same)
    run_len = np.bincount(run_id)
    long_runs = run_len[run_len >= ctx.config.max_stale_run]
    n_runs = int(long_runs.size)
    if n_runs == 0:
        return [
            _pass(
                "QP-DATA-011",
                "No stale prices",
                f"No run of ≥{ctx.config.max_stale_run} identical '{col}' values.",
            )
        ]
    return [
        Finding(
            id="QP-DATA-011",
            category=Category.DATA,
            severity=Severity.WARN,
            title="Suspicious forward-filled prices",
            message=(
                f"{n_runs} run(s) of at least {ctx.config.max_stale_run} identical consecutive "
                f"'{col}' values (longest: {int(np.max(long_runs))})."
            ),
            evidence={
                "runs": n_runs,
                "longest": int(np.max(long_runs)),
                "threshold": ctx.config.max_stale_run,
            },
            why_it_matters=(
                "Forward-filled prices create artificial zero returns, understate volatility, and "
                "can create fills at prices that never traded."
            ),
            recommendation="Mark non-trading periods explicitly instead of carrying prices forward.",
            confidence=Confidence.MEDIUM,
        )
    ]


def _check_infinite(ctx: _Context) -> list[Finding]:
    numeric = ctx.frame.select_dtypes(include=[np.number])
    n = int(np.isinf(numeric.to_numpy(dtype=float)).sum()) if numeric.size else 0
    if n == 0:
        return [_pass("QP-DATA-012", "No infinite values", "No infinite values found.")]
    return [
        Finding(
            id="QP-DATA-012",
            category=Category.DATA,
            severity=Severity.FAIL,
            title="Infinite values",
            message=f"{n} infinite value(s) found in numeric columns.",
            evidence={"count": n},
            why_it_matters="Infinite values poison every downstream statistic.",
            recommendation="Trace and fix the computation that produced them (often a divide by 0).",
            confidence=Confidence.HIGH,
        )
    ]


def _check_extreme_returns(ctx: _Context) -> list[Finding]:
    col = primary_price_column(ctx.frame.columns)
    if col is None:
        return []
    s = pd.Series(pd.to_numeric(ctx.frame[col], errors="coerce").to_numpy(), index=ctx.ts)
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.where(s > 0)
    rets = s.pct_change(fill_method=None)
    thr = ctx.config.extreme_return_threshold
    bad = rets.abs() > thr
    n = int(bad.sum())
    if n == 0:
        return [
            _pass(
                "QP-DATA-013",
                "No extreme returns",
                f"No single-period absolute return above {thr:.0%}.",
            )
        ]
    worst = rets.abs().idxmax()
    return [
        Finding(
            id="QP-DATA-013",
            category=Category.DATA,
            severity=Severity.WARN,
            title="Extreme single-period returns",
            message=(
                f"{n} period(s) have |return| > {thr:.0%}; largest {float(rets.loc[worst]):+.1%} "
                f"at {_fmt(worst)}."
            ),
            evidence={"count": n, "threshold": thr, "largest": float(rets.loc[worst])},
            location=_range(pd.DatetimeIndex(rets.index[bad.to_numpy()])),
            why_it_matters=(
                "Extreme moves are often unadjusted splits, bad ticks, or symbol changes, and can "
                "dominate backtest P&L."
            ),
            recommendation="Verify corporate-action adjustments and remove bad ticks.",
            confidence=Confidence.MEDIUM,
        )
    ]


DATA_RULES: dict[str, Callable[[_Context], list[Finding]]] = {
    "QP-DATA-001": _check_invalid_timestamps,
    "QP-DATA-002": _check_monotonic,
    "QP-DATA-003": _check_required_ohlc,
    "QP-DATA-004": _check_duplicate_timestamps,
    "QP-DATA-005": _check_duplicate_rows,
    "QP-DATA-006": _check_ohlc_relationships,
    "QP-DATA-007": _check_nonpositive,
    "QP-DATA-008": _check_nulls,
    "QP-DATA-009": _check_gaps,
    "QP-DATA-010": _check_timezone,
    "QP-DATA-011": _check_stale,
    "QP-DATA-012": _check_infinite,
    "QP-DATA-013": _check_extreme_returns,
}


def _empty_finding(n: int) -> Finding:
    return Finding(
        id="QP-DATA-014",
        category=Category.DATA,
        severity=Severity.FAIL,
        title="Empty or insufficient data",
        message=f"Only {n} observation(s) available; nothing can be validated or audited.",
        evidence={"rows": n},
        confidence=Confidence.HIGH,
    )


def _validate_single(
    frame: pd.DataFrame, cfg: DataValidationConfig, ts_col: str | None
) -> list[Finding]:
    if ts_col is None:
        parsed = parse_timestamps(frame.index)
        body = frame.reset_index(drop=True)
    else:
        parsed = parse_timestamps(frame[ts_col])
        body = frame.drop(columns=[ts_col]).reset_index(drop=True)
    ctx = _Context(frame=body, ts=parsed.values, parsed=parsed, config=cfg)
    findings: list[Finding] = []
    disabled = set(cfg.disabled_rules)
    for rule_id, check in DATA_RULES.items():
        if rule_id not in disabled:
            findings.extend(check(ctx))
    return findings


def _merge_by_rule(per_symbol: dict[str, list[Finding]]) -> list[Finding]:
    """Combine per-symbol findings into one finding per rule (worst severity wins)."""
    by_rule: dict[str, dict[str, Finding]] = {}
    for sym, fs in per_symbol.items():
        for f in fs:
            by_rule.setdefault(f.id, {})[sym] = f
    out: list[Finding] = []
    for items in by_rule.values():
        worst = max(items.values(), key=lambda f: f.severity.rank)
        affected = {s: f for s, f in items.items() if f.severity is not Severity.PASS}
        if not affected:
            out.append(
                worst.model_copy(update={"message": f"{worst.message} (all {len(items)} symbols)"})
            )
            continue
        shown = sorted(affected)[:5]
        more = len(affected) - len(shown)
        msg = (
            f"{len(affected)} of {len(items)} symbol(s) affected — "
            + "; ".join(f"{s}: {affected[s].message}" for s in shown)
            + (f"; … and {more} more" if more > 0 else "")
        )
        starts = [f.location.start for f in affected.values() if f.location and f.location.start]
        ends = [f.location.end for f in affected.values() if f.location and f.location.end]
        out.append(
            worst.model_copy(
                update={
                    "message": msg,
                    "evidence": {
                        "symbols_affected": sorted(affected),
                        "by_symbol": {s: f.evidence for s, f in affected.items()},
                    },
                    "location": Location(start=min(starts), end=max(ends))
                    if starts and ends
                    else None,
                }
            )
        )
    return out


def _sync_finding(raw: pd.DataFrame, ts_col: str, sym_col: str) -> Finding:
    ts = parse_timestamps(raw[ts_col]).values
    df = pd.DataFrame({"ts": ts, "sym": raw[sym_col].astype(str).to_numpy()}).dropna()
    sets = {s: set(g["ts"]) for s, g in df.groupby("sym")}
    union = set().union(*sets.values()) if sets else set()
    coverage = {s: len(v) / len(union) if union else 1.0 for s, v in sets.items()}
    spans = {s: (min(v), max(v)) for s, v in sets.items() if v}
    common_start = max(a for a, _ in spans.values()) if spans else None
    common_end = min(b for _, b in spans.values()) if spans else None
    inside = (
        {s: {t for t in v if common_start <= t <= common_end} for s, v in sets.items()}
        if spans
        else {}
    )
    inner_union = set().union(*inside.values()) if inside else set()
    gaps = {s: len(inner_union - v) for s, v in inside.items()}
    evidence = {
        "symbols": len(sets),
        "coverage": {s: round(c, 4) for s, c in coverage.items()},
        "missing_inside_common_span": gaps,
    }
    if any(gaps.values()):
        sev, msg = (
            Severity.WARN,
            (
                f"{sum(1 for g in gaps.values() if g)} symbol(s) miss timestamps that other symbols "
                "have inside the common date range; cross-sectional signals may compare prices from "
                "different moments."
            ),
        )
    elif len({spans[s] for s in spans}) > 1:
        sev, msg = (
            Severity.INFO,
            "Symbols start or end on different dates but are synchronized where they overlap.",
        )
    else:
        sev, msg = Severity.PASS, f"All {len(sets)} symbols share the same timestamps."
    return Finding(
        id="QP-DATA-015",
        category=Category.DATA,
        severity=sev,
        title="Symbols synchronized" if sev is Severity.PASS else "Unsynchronized symbols",
        message=msg,
        evidence=evidence,
        confidence=Confidence.HIGH,
    )


def validate_data(
    frame: pd.DataFrame,
    config: DataValidationConfig | None = None,
    *,
    timestamp_column: str | None = None,
    symbol_column: str | None = None,
) -> list[Finding]:
    """Run all enabled data-quality rules on a *raw* (unprepared) frame.

    Parameters
    ----------
    frame:
        Raw data as returned by :func:`quantproof.data.loaders.load_frame`, or any
        DataFrame with a DatetimeIndex or a timestamp column. Data with a symbol column (or a
        ``(timestamp, symbol)`` MultiIndex) is validated per symbol, and the results are
        merged per rule (worst severity, with a per-symbol breakdown in the evidence).
    config:
        Thresholds and disabled rules.
    timestamp_column, symbol_column:
        Names of the timestamp / symbol columns if they cannot be auto-detected.
    """
    cfg = config or DataValidationConfig()
    if not cfg.enabled:
        return []
    frame = flatten_multiindex(frame)
    if len(frame) == 0:
        return [_empty_finding(0)]
    sym_col = (
        normalize_column_name(symbol_column) if symbol_column else find_symbol_column(frame.columns)
    )
    ts_col = resolve_timestamp_column(frame, timestamp_column)
    if sym_col is None:
        return _validate_single(frame, cfg, ts_col)
    if ts_col is None:
        frame = frame.reset_index()
        ts_col = resolve_timestamp_column(frame, None)
        assert ts_col is not None
    per_symbol = {
        str(sym): _validate_single(group.drop(columns=[sym_col]), cfg, ts_col)
        for sym, group in frame.groupby(sym_col, sort=True)
    }
    findings = _merge_by_rule(per_symbol)
    if "QP-DATA-015" not in set(cfg.disabled_rules):
        findings.append(_sync_finding(frame, ts_col, sym_col))
    return findings
