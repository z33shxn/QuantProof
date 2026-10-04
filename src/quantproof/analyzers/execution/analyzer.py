"""Execution-realism audit: lag sensitivity, cost sensitivity, turnover, assumptions.

Scenarios simulated (same signals, different assumptions)
---------------------------------------------------------
``naive``      fill at the signal bar's close (lag 0), no costs — the most common
               optimistic backtest.
``declared``   the strategy's own ``EXECUTION`` assumptions (if declared).
``realistic``  the auditor's :class:`~quantproof.config.ExecutionConfig`.

Verdict rules (explicit)
------------------------
* ``QP-EXEC-001`` execution assumptions. The *headline* is the strategy's declared
  scenario (or the naive scenario if nothing is declared). The declaration is
  *optimistic* if it fills at the signal bar's close (``signal_lag=0``), declares no
  non-zero cost, or is absent.
  - FAIL if the declaration is optimistic **and** the headline Sharpe is ≥ 0.5 **and**
    the realistic net Sharpe retains less than 25 % of it.
  - WARN if ``signal_lag=0`` is declared but performance survives realistic execution.
  - otherwise PASS (with the lag-sensitivity table as evidence).
* ``QP-EXEC-002`` transaction costs: WARN if the strategy declares no non-zero cost
  assumption; INFO if declared one-way costs are below the auditor's; else PASS.
* ``QP-EXEC-003`` cost survival: WARN if realistic net Sharpe ≤ 0 while gross Sharpe > 0,
  or if the break-even cost multiplier (mean gross return / mean audit cost) is below 2,
  i.e. doubling the audit cost model would erase the mean return.
* ``QP-EXEC-004`` turnover: WARN if costs consume ≥ 50 % of gross return, else INFO.
* ``QP-EXEC-006`` WARN when the strategy declares an execution style (intrabar, event-driven,
  VWAP/TWAP) that cannot be simulated or verified from bar data.

Execution semantics
-------------------
``close``       close-to-close: decided at close *t*, filled at close *t + lag* (lag 0 = same bar).
``next_close``  as ``close`` with lag ≥ 1.
``next_open``   decided at close *t*, filled at the open of bar *t + lag*.
``limit``       limit order resting during bar *t + 1*; optimistic fill-on-touch.
Intrabar, event-driven, VWAP and TWAP fills are *not* simulated; they are reported as
untestable rather than silently approximated.

Cost attribution
----------------
Every cost component returns a per-bar cost as a fraction of equity, so gross return,
each component and net return are additive: ``net_t = gross_t − Σ_k cost_k,t``. The
attribution table reports these sums (arithmetic, over the evaluation window) and their
annualized means. The cost-multiplier table rescales the *whole* audit cost model
(including impact) by each multiplier; impact is not linear in size, so a multiplier is
a stress on cost *levels*, not a change of trade size.
"""

from __future__ import annotations

import math
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from quantproof.config import ExecutionConfig
from quantproof.errors import QuantProofInputError
from quantproof.execution.costs import TransactionCostModel
from quantproof.execution.fills import SimulationResult, simulate
from quantproof.results import Category, Finding
from quantproof.severity import Confidence, Severity
from quantproof.statistics.sharpe import annualized_return, max_drawdown, sharpe_ratio

_T = TypeVar("_T", pd.Series, pd.DataFrame)

EDGE_RETAINED_FAIL = 0.25
BREAK_EVEN_MULTIPLIER_WARN = 2.0

EXECUTION_SEMANTICS: dict[str, str] = {
    "close": "close-to-close: decided at close t, filled at close t+lag",
    "next_close": "decided at close t, filled at close t+lag (lag >= 1)",
    "next_open": "decided at close t, filled at the open of bar t+lag",
    "limit": "limit order resting during bar t+1, filled on touch (optimistic)",
}
UNTESTABLE_FILLS: dict[str, str] = {
    "intrabar": "fills inside a bar at prices bar data does not reveal",
    "event": "event-driven fills at arbitrary times",
    "event_driven": "event-driven fills at arbitrary times",
    "vwap": "volume-weighted fills over an interval",
    "twap": "time-weighted fills over an interval",
}
MIN_NAIVE_SHARPE = 0.5
COST_DRAG_WARN = 0.5


def metrics(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    """Headline metrics of a return series."""
    r = returns.dropna()
    return {
        "sharpe": sharpe_ratio(r, periods_per_year=periods_per_year),
        "cagr": annualized_return(r, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(r),
        "volatility": float(r.std(ddof=1) * math.sqrt(periods_per_year))
        if len(r) > 1
        else float("nan"),
        "total_return": float(np.prod(1 + r.to_numpy()) - 1) if len(r) else float("nan"),
    }


def cost_model_from(cfg: ExecutionConfig | dict[str, Any]) -> TransactionCostModel:
    """Linear cost model from an ExecutionConfig or a declared EXECUTION dict."""
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=0.0: getattr(cfg, k, d))
    return TransactionCostModel.from_bps(
        commission_bps=float(get("commission_bps", 0.0) or 0.0),
        spread_bps=float(get("spread_bps", 0.0) or 0.0),
        slippage_bps=float(get("slippage_bps", 0.0) or 0.0),
        impact_coefficient=float(get("impact_coefficient", 0.0) or 0.0),
        tax_bps=float(get("tax_bps", 0.0) or 0.0),
    )


def one_way_bps(d: dict[str, Any]) -> float:
    """Linear one-way cost (commission + half spread + slippage + taxes) in bps."""
    return (
        float(d.get("commission_bps", 0) or 0)
        + float(d.get("spread_bps", 0) or 0) / 2
        + float(d.get("slippage_bps", 0) or 0)
        + float(d.get("tax_bps", 0) or 0)
    )


def declared_fill(declared: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """``(fill, untestable_reason)`` for a declared EXECUTION dict; validates the name."""
    if declared is None:
        return None, None
    fill = str(declared.get("fill", "close")).strip().lower().replace("-", "_")
    if fill in UNTESTABLE_FILLS:
        return fill, UNTESTABLE_FILLS[fill]
    if fill not in EXECUTION_SEMANTICS:
        raise QuantProofInputError(
            f"EXECUTION['fill'] = {declared.get('fill')!r} is not recognised. Use one of "
            f"{sorted(EXECUTION_SEMANTICS)} (simulated) or {sorted(UNTESTABLE_FILLS)} "
            "(declared but not testable from bars)."
        )
    return fill, None


def cost_attribution(
    gross: pd.Series, breakdown: pd.DataFrame, periods_per_year: float
) -> dict[str, Any]:
    """Additive decomposition gross − Σ components = net (sums and annualized means)."""
    comps = breakdown.drop(columns=["none"], errors="ignore")
    total = comps.sum(axis=1) if comps.shape[1] else pd.Series(0.0, index=gross.index)
    net = gross - total.reindex(gross.index, fill_value=0.0)
    n = max(len(gross), 1)

    def ann(x: float) -> float:
        return float(x / n * periods_per_year)

    rows = [{"item": "gross", "sum": float(gross.sum()), "annualized": ann(gross.sum())}]
    for c in comps.columns:
        v = -float(comps[c].sum())
        rows.append({"item": str(c), "sum": v, "annualized": ann(v)})
    rows.append({"item": "net", "sum": float(net.sum()), "annualized": ann(net.sum())})
    return {
        "rows": rows,
        "total_costs": float(total.sum()),
        "units": "sum of per-bar returns, fraction of equity (arithmetic, not compounded)",
    }


def realistic_simulation(
    prices: pd.DataFrame,
    signals: pd.Series | pd.DataFrame,
    cfg: ExecutionConfig,
    periods_per_year: float,
) -> SimulationResult:
    """Simulate under the auditor's assumptions (used for statistics downstream)."""
    fill = cfg.fill if cfg.fill != "next_open" or "open" in prices.columns else "close"
    return simulate(
        prices,
        signals,
        fill=fill,
        lag=cfg.signal_lag,
        cost_model=cost_model_from(cfg),
        capital=cfg.capital,
        periods_per_year=periods_per_year,
    )


def analyze_execution(
    prices: pd.DataFrame,
    signals: pd.Series | pd.DataFrame,
    cfg: ExecutionConfig,
    *,
    declared: dict[str, Any] | None,
    periods_per_year: float = 252.0,
    evaluation_start: Any = None,
) -> tuple[dict[str, Any], list[Finding], SimulationResult]:
    """Run the execution scenarios and produce findings.

    Returns ``(section, findings, realistic_simulation)``.
    """

    def window(s: _T) -> _T:
        return s.loc[evaluation_start:] if evaluation_start is not None else s

    findings: list[Finding] = []
    naive = simulate(prices, signals, fill="close", lag=0, periods_per_year=periods_per_year)
    realistic = realistic_simulation(prices, signals, cfg, periods_per_year)
    scenarios: dict[str, Any] = {
        "naive": {
            "assumptions": {"signal_lag": 0, "fill": "close", "one_way_cost_bps": 0.0},
            "metrics": metrics(window(naive.net_returns), periods_per_year),
        },
        "realistic": {
            "assumptions": {
                "signal_lag": cfg.signal_lag,
                "fill": cfg.fill,
                "commission_bps": cfg.commission_bps,
                "spread_bps": cfg.spread_bps,
                "slippage_bps": cfg.slippage_bps,
                "impact_coefficient": cfg.impact_coefficient,
                "one_way_cost_bps": cfg.one_way_cost_bps,
            },
            "metrics": metrics(window(realistic.net_returns), periods_per_year),
            "gross_metrics": metrics(window(realistic.gross_returns), periods_per_year),
        },
    }
    d_fill, untestable = declared_fill(declared)
    if declared is not None and untestable is None:
        d_sim = simulate(
            prices,
            signals,
            fill=str(d_fill),
            lag=int(declared.get("signal_lag", 1)),
            cost_model=cost_model_from(declared),
            capital=float(declared.get("capital", cfg.capital)),
            periods_per_year=periods_per_year,
        )
        scenarios["declared"] = {
            "assumptions": {**declared, "one_way_cost_bps": one_way_bps(declared)},
            "metrics": metrics(window(d_sim.net_returns), periods_per_year),
        }

    # Lag sensitivity (gross, close fills)
    lag_rows: list[dict[str, Any]] = []
    for lag in (0, 1, 2, 5):
        sim = simulate(prices, signals, fill="close", lag=lag, periods_per_year=periods_per_year)
        lag_rows.append({"signal_lag": lag, **metrics(window(sim.gross_returns), periods_per_year)})
    if "open" in prices.columns:
        sim = simulate(prices, signals, fill="next_open", lag=1, periods_per_year=periods_per_year)
        lag_rows.append(
            {"signal_lag": "next_open", **metrics(window(sim.gross_returns), periods_per_year)}
        )

    # Cost sensitivity: the whole audit cost model scaled by each multiplier.
    gross = window(realistic.gross_returns)
    costs = window(realistic.costs)
    turnover = window(realistic.gross_turnover)
    cost_rows = []
    for m in cfg.cost_multipliers:
        net = gross - costs * m
        cost_rows.append(
            {
                "multiplier": float(m),
                "linear_one_way_cost_bps": float(m) * cfg.one_way_cost_bps,
                **metrics(net, periods_per_year),
            }
        )
    mean_turnover = float(turnover.mean()) if len(turnover) else 0.0
    mean_gross = float(gross.mean()) if len(gross) else float("nan")
    mean_cost = float(costs.mean()) if len(costs) else 0.0
    # Break-even is only defined for a positive mean gross return (otherwise no cost level
    # makes the strategy profitable); report NaN rather than a negative "break-even".
    if not (math.isfinite(mean_gross) and mean_gross > 0):
        break_even = break_even_mult = float("nan")
    else:
        break_even = mean_gross / mean_turnover * 1e4 if mean_turnover > 0 else float("inf")
        break_even_mult = mean_gross / mean_cost if mean_cost > 0 else float("inf")
    attribution = cost_attribution(gross, window(realistic.cost_breakdown), periods_per_year)

    sr_naive = lag_rows[0]["sharpe"]
    sr_lag1 = lag_rows[1]["sharpe"]
    edge_retained = (
        sr_lag1 / sr_naive
        if math.isfinite(sr_naive) and sr_naive > 0 and math.isfinite(sr_lag1)
        else float("nan")
    )
    declared_lag = None if declared is None else declared.get("signal_lag")
    declared_cost = one_way_bps(declared) if declared else 0.0
    optimistic = declared is None or declared_lag == 0 or declared_cost == 0
    headline_key = "declared" if "declared" in scenarios else "naive"
    sr_headline = scenarios[headline_key]["metrics"]["sharpe"]
    sr_real = scenarios["realistic"]["metrics"]["sharpe"]
    retained = (
        sr_real / sr_headline
        if math.isfinite(sr_headline) and sr_headline > 0 and math.isfinite(sr_real)
        else float("nan")
    )
    section: dict[str, Any] = {
        "scenarios": scenarios,
        "lag_sensitivity": lag_rows,
        "cost_sensitivity": cost_rows,
        "cost_attribution": attribution,
        "break_even_one_way_cost_bps": break_even,
        "break_even_cost_multiplier": break_even_mult,
        "semantics": {
            "audit": EXECUTION_SEMANTICS.get(cfg.fill, cfg.fill),
            "declared": None
            if d_fill is None
            else EXECUTION_SEMANTICS.get(d_fill, UNTESTABLE_FILLS.get(d_fill, d_fill)),
            "declared_simulated": declared is not None and untestable is None,
        },
        "edge_retained_after_one_bar_lag": edge_retained,
        "headline_scenario": headline_key,
        "headline_retained_under_audit": retained,
        "turnover": realistic.turnover,
        "declared_assumptions": declared,
        "audit_assumptions": cfg.model_dump(),
        "cost_breakdown_total": {
            k: float(v) for k, v in window(realistic.cost_breakdown).sum().items()
        },
        "fill_model": realistic.fill_model,
        "limitations": [
            "Fills are simulated at bar prices without queue position, partial fills or "
            "latency within the bar.",
            "Notional-dependent costs assume constant capital.",
            "Break-even one-way cost assumes linear costs; the break-even multiplier scales the "
            "full model including impact.",
            "The square-root impact model is an order-of-magnitude approximation.",
        ],
    }

    # QP-EXEC-001
    ev = {
        "headline_scenario": headline_key,
        "headline_sharpe": sr_headline,
        "lag0_gross_sharpe": sr_naive,
        "lag1_gross_sharpe": sr_lag1,
        "realistic_net_sharpe": sr_real,
        "retained": retained,
        "declared_signal_lag": declared_lag,
        "declared_one_way_bps": declared_cost,
        "thresholds": {
            "retained_fail": EDGE_RETAINED_FAIL,
            "min_headline_sharpe": MIN_NAIVE_SHARPE,
        },
    }
    collapses = (
        math.isfinite(sr_headline)
        and sr_headline >= MIN_NAIVE_SHARPE
        and (not math.isfinite(retained) or retained < EDGE_RETAINED_FAIL)
    )
    if optimistic and collapses:
        reasons = []
        if declared is None:
            reasons.append("no execution assumptions declared")
        if declared_lag == 0:
            reasons.append("fills at the signal bar's close")
        if declared is not None and declared_cost == 0:
            reasons.append("zero transaction costs")
        findings.append(
            Finding(
                id="QP-EXEC-001",
                category=Category.EXECUTION,
                severity=Severity.FAIL,
                title="Performance depends on unrealistic execution assumptions",
                message=(
                    f"Headline Sharpe {sr_headline:.2f} ({'; '.join(reasons)}) falls to "
                    f"{sr_real:.2f} under audit assumptions (lag {cfg.signal_lag}, "
                    f"{cfg.one_way_cost_bps:.1f} bps one-way): "
                    + (f"{retained:.0%}" if math.isfinite(retained) else "none")
                    + f" of the edge survives (< {EDGE_RETAINED_FAIL:.0%}). Gross Sharpe at lag 0 / "
                    f"lag 1: {sr_naive:.2f} / {sr_lag1:.2f}."
                ),
                evidence=ev,
                why_it_matters=(
                    "Filling at the exact close used to compute the signal requires zero latency, "
                    "and costs are never zero. If the edge disappears under a one-bar delay and "
                    "modest costs, the reported performance is an artefact of the assumptions."
                ),
                recommendation=(
                    "Evaluate with signal_lag ≥ 1 (or next-open fills) and realistic costs, and "
                    "report those results."
                ),
                confidence=Confidence.HIGH,
            )
        )
    elif declared_lag == 0:
        findings.append(
            Finding(
                id="QP-EXEC-001",
                category=Category.EXECUTION,
                severity=Severity.WARN,
                title="Same-bar execution assumption detected",
                message=(
                    "The strategy declares signal_lag=0 (fill at the close that produced the "
                    f"signal). Gross Sharpe at lag 0 / lag 1: {sr_naive:.2f} / {sr_lag1:.2f}; "
                    f"realistic net Sharpe {sr_real:.2f}."
                ),
                evidence=ev,
                why_it_matters="Zero-latency closing fills are rarely achievable in practice.",
                recommendation="Report results with at least one bar of execution lag.",
                confidence=Confidence.HIGH,
            )
        )
    else:
        findings.append(
            Finding(
                id="QP-EXEC-001",
                category=Category.EXECUTION,
                severity=Severity.PASS,
                title="Edge does not depend on optimistic execution assumptions",
                message=(
                    f"Headline ({headline_key}) Sharpe {sr_headline:.2f}; realistic net Sharpe "
                    f"{sr_real:.2f}; gross Sharpe at lag 0 / lag 1: {sr_naive:.2f} / {sr_lag1:.2f}."
                ),
                evidence=ev,
            )
        )

    # QP-EXEC-002
    if declared is None or declared_cost == 0:
        findings.append(
            Finding(
                id="QP-EXEC-002",
                category=Category.EXECUTION,
                severity=Severity.WARN,
                title="Transaction cost model missing",
                message=(
                    "The strategy declares no transaction-cost assumptions; QuantProof applied "
                    f"{cfg.one_way_cost_bps:.1f} bps one-way (commission {cfg.commission_bps}, "
                    f"half-spread {cfg.spread_bps / 2}, slippage {cfg.slippage_bps})."
                ),
                evidence={
                    "declared_one_way_bps": declared_cost,
                    "audit_one_way_bps": cfg.one_way_cost_bps,
                },
                why_it_matters="Gross results overstate what a strategy can earn.",
                recommendation="Declare cost assumptions in EXECUTION and justify them.",
                confidence=Confidence.HIGH,
            )
        )
    elif declared_cost < cfg.one_way_cost_bps:
        findings.append(
            Finding(
                id="QP-EXEC-002",
                category=Category.EXECUTION,
                severity=Severity.INFO,
                title="Declared costs below audit assumptions",
                message=(
                    f"Declared one-way cost {declared_cost:.1f} bps < audit assumption "
                    f"{cfg.one_way_cost_bps:.1f} bps."
                ),
                evidence={
                    "declared_one_way_bps": declared_cost,
                    "audit_one_way_bps": cfg.one_way_cost_bps,
                },
            )
        )
    else:
        findings.append(
            Finding(
                id="QP-EXEC-002",
                category=Category.EXECUTION,
                severity=Severity.PASS,
                title="Transaction costs declared",
                message=f"Declared one-way cost: {declared_cost:.1f} bps.",
                evidence={"declared_one_way_bps": declared_cost},
            )
        )

    # QP-EXEC-003
    gross_sr = scenarios["realistic"]["gross_metrics"]["sharpe"]
    net_sr = scenarios["realistic"]["metrics"]["sharpe"]
    margin_ok = break_even_mult >= BREAK_EVEN_MULTIPLIER_WARN
    if (
        math.isfinite(gross_sr)
        and gross_sr > 0
        and (not math.isfinite(net_sr) or net_sr <= 0 or not margin_ok)
    ):
        findings.append(
            Finding(
                id="QP-EXEC-003",
                category=Category.EXECUTION,
                severity=Severity.WARN,
                title="Edge is fragile to transaction costs",
                message=(
                    f"Gross Sharpe {gross_sr:.2f} vs net {net_sr:.2f} at {cfg.one_way_cost_bps:.1f} "
                    f"bps one-way. Mean return is erased at {break_even_mult:.2f}× the audit cost "
                    f"model (break-even linear cost ≈ {break_even:.1f} bps one-way)."
                ),
                evidence={
                    "gross_sharpe": gross_sr,
                    "net_sharpe": net_sr,
                    "break_even_bps": break_even,
                    "break_even_multiplier": break_even_mult,
                    "threshold_multiplier": BREAK_EVEN_MULTIPLIER_WARN,
                    "audit_one_way_bps": cfg.one_way_cost_bps,
                },
                why_it_matters="Small errors in cost estimates would eliminate the strategy's return.",
                recommendation="Reduce turnover or verify costs with real execution data.",
                confidence=Confidence.HIGH,
            )
        )
    else:
        findings.append(
            Finding(
                id="QP-EXEC-003",
                category=Category.EXECUTION,
                severity=Severity.PASS if math.isfinite(net_sr) and net_sr > 0 else Severity.INFO,
                title="Edge survives transaction costs"
                if math.isfinite(net_sr) and net_sr > 0
                else "No gross edge to erode",
                message=(
                    f"Break-even at {break_even_mult:.2f}× the audit cost model (linear "
                    f"≈ {break_even:.1f} bps one-way vs {cfg.one_way_cost_bps:.1f} bps assumed; "
                    f"net Sharpe {net_sr:.2f})."
                    if math.isfinite(break_even_mult)
                    else f"The mean gross return is not positive, so no cost level breaks even "
                    f"(gross Sharpe {gross_sr:.2f}, net Sharpe {net_sr:.2f})."
                ),
                evidence={
                    "break_even_bps": break_even,
                    "break_even_multiplier": break_even_mult,
                    "net_sharpe": net_sr,
                    "gross_sharpe": gross_sr,
                },
            )
        )

    # QP-EXEC-004
    gross_total = float(gross.sum())
    cost_total = float(costs.sum())
    drag = cost_total / gross_total if gross_total > 0 else float("nan")
    turn = realistic.turnover.get("annualized", float("nan"))
    heavy = math.isfinite(drag) and drag >= COST_DRAG_WARN
    findings.append(
        Finding(
            id="QP-EXEC-004",
            category=Category.EXECUTION,
            severity=Severity.WARN if heavy else Severity.INFO,
            title="Excessive turnover relative to edge" if heavy else "Turnover",
            message=(
                f"Annualized turnover {turn:.1f}× equity; costs consume "
                + (f"{drag:.0%}" if math.isfinite(drag) else "n/a")
                + " of gross return."
            ),
            evidence={"annualized_turnover": turn, "cost_drag": drag, "threshold": COST_DRAG_WARN},
            why_it_matters="High turnover multiplies the impact of any cost misestimate.",
            recommendation="Add trading buffers or lower rebalance frequency.",
            confidence=Confidence.MEDIUM,
        )
    )

    # QP-EXEC-006
    if untestable is not None:
        findings.append(
            Finding(
                id="QP-EXEC-006",
                category=Category.EXECUTION,
                severity=Severity.WARN,
                title="Execution semantics not testable from bars",
                message=(
                    f"The strategy declares fill={d_fill!r} ({untestable}). Bar data cannot "
                    "verify these fills, so the declared scenario was not simulated; the audit "
                    f"used {EXECUTION_SEMANTICS.get(cfg.fill, cfg.fill)} instead."
                ),
                evidence={"declared_fill": d_fill, "audit_fill": cfg.fill},
                confidence=Confidence.HIGH,
            )
        )
    return section, findings, realistic
