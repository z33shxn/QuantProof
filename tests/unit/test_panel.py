"""Multi-asset (panel) data: preparation, validation, simulation, causality, audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantproof import AuditConfig, audit
from quantproof.analyzers.causal import perturb_after, run_causality_test
from quantproof.config import CausalityConfig
from quantproof.data.loaders import prepare_prices
from quantproof.data.panel import is_panel, symbols, to_wide_signals, unique_times
from quantproof.data.synthetic import generate_prices
from quantproof.data.validation import validate_data
from quantproof.errors import QuantProofDataError
from quantproof.execution import TransactionCostModel, simulate
from quantproof.severity import Severity

SYMS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    frames = {s: generate_prices(300, seed=i) for i, s in enumerate(SYMS)}
    return pd.concat(frames, names=["symbol"]).swaplevel().sort_index()


@pytest.fixture(scope="module")
def long(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.reset_index()


def momentum(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    close = data["close"].unstack()
    rank = close.pct_change(lookback).rank(axis=1)
    w = rank.sub(rank.mean(axis=1), axis=0)
    return w.div(w.abs().sum(axis=1), axis=0)


def peeking(data: pd.DataFrame) -> pd.DataFrame:
    close = data["close"].unstack()
    rank = close.pct_change().shift(-1).rank(axis=1)
    w = rank.sub(rank.mean(axis=1), axis=0)
    return w.div(w.abs().sum(axis=1), axis=0)


def test_long_format_becomes_sorted_panel(long: pd.DataFrame) -> None:
    shuffled = long.sample(frac=1.0, random_state=0)
    prepared = prepare_prices(shuffled)
    assert is_panel(prepared.prices)
    assert symbols(prepared.prices) == SYMS
    assert prepared.prices.index.is_monotonic_increasing
    assert any("Sorted" in a for a in prepared.actions)


def test_duplicate_symbol_rows_are_removed_and_logged(long: pd.DataFrame) -> None:
    dup = pd.concat([long, long.iloc[:3]])
    prepared = prepare_prices(dup)
    assert len(prepared.prices) == len(long)
    assert any("duplicated (timestamp, symbol)" in a for a in prepared.actions)


def test_single_timestamp_panel_is_rejected(long: pd.DataFrame) -> None:
    first = long[long["timestamp"] == long["timestamp"].min()]
    with pytest.raises(QuantProofDataError, match="at least 2 distinct timestamps"):
        prepare_prices(first)


def test_validation_merges_per_symbol_findings(long: pd.DataFrame) -> None:
    broken = long.copy()
    rows = broken.index[broken["symbol"] == "BBB"][::50]
    broken.loc[rows, "close"] = -1.0
    issues = [f for f in validate_data(broken) if f.is_issue]
    named = [f for f in issues if "BBB" in str(f.evidence)]
    assert named, "an issue in one symbol must be reported with the symbol named"
    assert all("AAA" not in str(f.evidence.get("symbols_affected", "")) for f in named)


def test_unsynchronized_symbols_are_reported(long: pd.DataFrame) -> None:
    gappy = long[~((long["symbol"] == "CCC") & (long.index % 3 == 0))]
    ids = {f.id: f.severity for f in validate_data(gappy)}
    assert ids.get("QP-DATA-015") is Severity.WARN


def test_to_wide_signals_rejects_unknown_symbols(panel: pd.DataFrame) -> None:
    sig = momentum(panel).rename(columns={"AAA": "ZZZ"})
    with pytest.raises(QuantProofDataError, match="symbols not in the data"):
        to_wide_signals(sig, panel)
    stacked = momentum(panel).stack()  # noqa: PD013 - MultiIndex output under test
    out = to_wide_signals(stacked, panel)
    assert list(out.columns) == SYMS and len(out) == len(unique_times(panel))


@pytest.mark.parametrize(("lag", "shift"), [(0, 1), (1, 2), (2, 3)])
def test_portfolio_gross_returns_are_exact(panel: pd.DataFrame, lag: int, shift: int) -> None:
    w = momentum(panel)
    res = simulate(panel, w, fill="close", lag=lag)
    rets = panel["close"].unstack().pct_change()
    manual = (w.shift(shift).fillna(0.0) * rets.fillna(0.0)).sum(axis=1)
    np.testing.assert_allclose(res.gross_returns.to_numpy(), manual.to_numpy(), atol=1e-12)


def test_portfolio_costs_sum_over_symbols(panel: pd.DataFrame) -> None:
    w = momentum(panel)
    model = TransactionCostModel.from_bps(commission_bps=1, spread_bps=2, slippage_bps=2)
    res = simulate(panel, w, fill="close", lag=1, cost_model=model)
    # 1 + 2/2 + 2 = 4 bps one-way on every unit of traded weight.
    np.testing.assert_allclose(res.costs.sum(), res.gross_turnover.sum() * 4e-4, rtol=1e-9)
    np.testing.assert_allclose(res.net_returns, res.gross_returns - res.costs)
    assert res.symbol_gross is not None and list(res.symbol_gross.columns) == SYMS


def test_panel_perturbation_is_per_symbol_and_causal(panel: pd.DataFrame) -> None:
    times = unique_times(panel)
    cutoff = times[150]
    for scheme in ("additive", "multiplicative", "permutation", "replacement", "extreme"):
        out = perturb_after(panel, cutoff, scheme, seed=0)
        past = panel.index.get_level_values(0) <= cutoff
        assert out.loc[past].equals(panel.loc[past].astype(float))
        assert not out.loc[~past].equals(panel.loc[~past].astype(float))
        if scheme in ("permutation", "replacement"):
            # values never move across symbols
            for s in SYMS:
                moved = set(out.xs(s, level=1)["close"].round(10))
                original = set(panel.xs(s, level=1)["close"].round(10))
                assert moved <= original


def test_cross_sectional_rank_is_not_a_false_positive(panel: pd.DataFrame) -> None:
    rep = run_causality_test(momentum, panel, CausalityConfig(n_timestamps=8))
    assert rep.n_fail == 0 and rep.n_symbols == len(SYMS)


def test_cross_asset_lookahead_is_detected(panel: pd.DataFrame) -> None:
    rep = run_causality_test(peeking, panel, CausalityConfig(n_timestamps=8))
    assert rep.n_fail > 0


def test_panel_audit_end_to_end(long: pd.DataFrame) -> None:
    cfg = AuditConfig(profile="quick")
    clean = audit(momentum, long, config=cfg)
    assert "QP-CAUSAL-001" not in clean.ids(Severity.FAIL)
    assert clean.sections["overview"]["n_symbols"] == len(SYMS)
    assert clean.sections["data"]["symbols"] == SYMS
    attribution = clean.sections["execution"]["cost_attribution"]["rows"]
    items = [r["item"] for r in attribution]
    assert items[0] == "gross" and items[-1] == "net"
    total = sum(r["sum"] for r in attribution[:-1])
    assert total == pytest.approx(attribution[-1]["sum"], abs=1e-12)

    bad = audit(peeking, long, config=cfg)
    assert {"QP001", "QP-CAUSAL-001"} <= set(bad.ids(Severity.FAIL))


def test_constant_prices_do_not_crash() -> None:
    idx = pd.bdate_range("2021-01-01", periods=200)
    df = pd.DataFrame(
        {"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 1_000}, index=idx
    )
    res = audit(
        lambda d: (d["close"].pct_change(5) > 0).astype(float),
        df,
        config=AuditConfig(profile="quick"),
    )
    assert "QP-DATA-011" in res.ids(Severity.WARN)
