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
  or if the break-even one-way cost is below twice the realistic one-way cost.
* ``QP-EXEC-004`` turnover: WARN if costs consume ≥ 50 % of gross return, else INFO.
"""

from __future__ import annotations

import math
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from quantproof.config import ExecutionConfig
from quantproof.execution.costs import TransactionCostModel
from quantproof.execution.fills import SimulationResult, simulate
from quantproof.results import Category, Finding
from quantproof.severity import Confidence, Severity
from quantproof.statistics.sharpe import annualized_return, max_drawdown, sharpe_ratio

_T = TypeVar("_T", pd.Series, pd.DataFrame)

EDGE_RETAINED_FAIL = 0.25
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
    )


def one_way_bps(d: dict[str, Any]) -> float:
    return (
        float(d.get("commission_bps", 0) or 0)
        + float(d.get("spread_bps", 0) or 0) / 2
        + float(d.get("slippage_bps", 0) or 0)
    )


def realistic_simulation(
    prices: pd.DataFrame, signals: pd.Series, cfg: ExecutionConfig, periods_per_year: float
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
    signals: pd.Series,
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
    if declared is not None:
        d_sim = simulate(
            prices,
            signals,
            fill=str(declared.get("fill", "close")),
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

    # Cost sensitivity at the realistic lag
    gross = window(realistic.gross_returns)
    turnover = window(realistic.trades).abs()
    cost_rows = []
    for bps in cfg.cost_grid_bps:
        net = gross - turnover * bps / 1e4
        cost_rows.append({"one_way_cost_bps": bps, **metrics(net, periods_per_year)})
    mean_turnover = float(turnover.mean())
    mean_gross = float(gross.mean())
    break_even = mean_gross / mean_turnover * 1e4 if mean_turnover > 0 else float("inf")

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
    headline_key = "declared" if declared is not None else "naive"
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
        "break_even_one_way_cost_bps": break_even,
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
    margin_ok = break_even >= 2 * cfg.one_way_cost_bps
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
                    f"bps one-way; break-even cost is {break_even:.1f} bps one-way."
                ),
                evidence={
                    "gross_sharpe": gross_sr,
                    "net_sharpe": net_sr,
                    "break_even_bps": break_even,
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
                    f"Break-even one-way cost {break_even:.1f} bps vs audit assumption "
                    f"{cfg.one_way_cost_bps:.1f} bps (net Sharpe {net_sr:.2f})."
                ),
                evidence={
                    "break_even_bps": break_even,
                    "net_sharpe": net_sr,
                    "gross_sharpe": gross_sr,
                },
            )
        )

    # QP-EXEC-004
    gross_total = float(gross.sum())
    cost_total = float(window(realistic.costs).sum())
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
    return section, findings, realistic
