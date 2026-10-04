"""Cost components, fill models, simulator, providers, turnover, timing."""

from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pandas as pd
import pytest

from quantproof.errors import QuantProofDataError, QuantProofInputError
from quantproof.execution import (
    BpsCommission,
    CostContext,
    FeeSchedule,
    FixedPerOrderCommission,
    FixedSlippage,
    FixedSpread,
    LimitOrder,
    MarketOnClose,
    NextOpen,
    PercentageCommission,
    PerShareCommission,
    ProviderCost,
    SizeSlippage,
    SquareRootImpact,
    TransactionCostModel,
    VolatilitySlippage,
    analyze_execution_timing,
    india_nse_provider,
    roll_spread,
    simulate,
    turnover_stats,
)
from quantproof.execution.fills import causal_volatility, make_fill_model
from quantproof.execution.providers import CostProvider


def ctx(trades, price=100.0, capital=1_000_000.0, vol=None, adv=None):
    n = len(trades)
    return CostContext(
        trades=np.asarray(trades, dtype=float),
        prices=np.full(n, price),
        capital=capital,
        timestamps=pd.date_range("2025-01-01", periods=n, freq="D"),
        volatility=None if vol is None else np.full(n, vol),
        adv=None if adv is None else np.full(n, adv),
    )


def test_bps_and_percentage_commission():
    c = ctx([0, 1, 0, -2])
    assert np.allclose(BpsCommission(10).cost(c), [0, 0.001, 0, 0.002])
    assert np.allclose(PercentageCommission(0.001).cost(c), [0, 0.001, 0, 0.002])


def test_per_share_and_fixed_commission():
    c = ctx([0.0, 0.5, 0.0001])
    # 0.5 * 1e6 / 100 = 5000 shares * $0.01 = $50 → 5e-5; small trade hits $1 minimum
    assert np.allclose(PerShareCommission(0.01, min_per_order=1.0).cost(c), [0, 5e-5, 1e-6])
    assert np.allclose(FixedPerOrderCommission(5.0).cost(c), [0, 5e-6, 5e-6])


def test_spread_slippage_impact():
    c = ctx([1.0, -0.5], vol=0.02, adv=10_000.0)
    assert np.allclose(FixedSpread(4).cost(c), [0.0002, 0.0001])
    assert np.allclose(FixedSlippage(3).cost(c), [0.0003, 0.00015])
    assert np.allclose(VolatilitySlippage(0.5).cost(c), [0.01, 0.005])
    # participation = shares / adv = 10000/10000 = 1 and 0.5
    assert np.allclose(SizeSlippage(10).cost(c), [1.0 * 10 * 1.0 / 1e4, 0.5 * 10 * 0.5 / 1e4])
    assert np.allclose(SquareRootImpact(1.0).cost(c), [0.02 * 1.0, 0.5 * 0.02 * np.sqrt(0.5)])


def test_components_require_inputs_and_validate_params():
    with pytest.raises(QuantProofInputError):
        VolatilitySlippage().cost(ctx([1]))
    with pytest.raises(QuantProofInputError):
        SquareRootImpact().cost(ctx([1], vol=0.01))
    with pytest.raises(QuantProofInputError):
        SizeSlippage().cost(ctx([1]))
    with pytest.raises(QuantProofInputError):
        BpsCommission(-1)


def test_cost_model_breakdown_and_describe():
    model = TransactionCostModel.from_bps(commission_bps=1, spread_bps=2, slippage_bps=3)
    bd = model.breakdown(ctx([0, 1]))
    assert list(bd.columns) == ["commission", "spread", "slippage"]
    assert model.total(ctx([0, 1]))[1] == pytest.approx((1 + 1 + 3) / 1e4)
    assert TransactionCostModel().is_zero
    assert len(model.describe()) == 3


@pytest.fixture
def tiny():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "open": [100, 101, 103, 102, 104.0],
            "high": [101, 103, 104, 103, 106.0],
            "low": [99, 100, 101, 100, 103.0],
            "close": [100, 102, 103, 101, 105.0],
        },
        index=idx,
    )


def test_market_on_close_lags(tiny):
    targets = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=tiny.index)
    r = tiny["close"].pct_change().fillna(0)
    lag0 = MarketOnClose(0).fill(tiny, targets)
    # held over bar s = targets[s-1]
    assert np.allclose(lag0.gross_returns, [0, r.iloc[1], r.iloc[2], 0, 0])
    lag1 = MarketOnClose(1).fill(tiny, targets)
    assert np.allclose(lag1.gross_returns, [0, 0, r.iloc[2], r.iloc[3], 0])
    assert lag1.trades.tolist() == [0.0, 1.0, 0.0, -1.0, 0.0]


def test_next_open_exact_compounding(tiny):
    targets = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=tiny.index)
    out = NextOpen(1).fill(tiny, targets)
    # bar 1: enter at open 101, earn 101→102 intraday
    assert out.gross_returns.iloc[1] == pytest.approx(102 / 101 - 1)
    # bar 3: exit at open 102: earn overnight 103→102, then flat
    assert out.gross_returns.iloc[3] == pytest.approx(102 / 103 - 1)
    with pytest.raises(QuantProofInputError):
        NextOpen(0)
    with pytest.raises(QuantProofDataError):
        NextOpen(1).fill(tiny[["close"]], targets)


def test_limit_order_fill_and_miss(tiny):
    targets = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=tiny.index)
    out = LimitOrder(offset_bps=50).fill(tiny, targets)
    # buy limit at 100*(1-0.005)=99.5; bar 1 low=100 → not filled; bar 2 limit 101.49, low 101 → fill
    assert out.weights.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert out.exec_prices.iloc[2] == pytest.approx(min(102 * 0.995, 103))
    assert out.fill_rate == pytest.approx(0.5)


def test_make_fill_model():
    assert isinstance(make_fill_model("close", 2), MarketOnClose)
    assert isinstance(make_fill_model("next_open", 0), NextOpen)
    assert isinstance(make_fill_model("limit", 1), LimitOrder)
    with pytest.raises(QuantProofInputError):
        make_fill_model("vwap", 1)


def test_simulate_net_equals_gross_minus_costs(prices):
    sig = np.sign(prices["close"].diff(5)).fillna(0)
    res = simulate(prices, sig, lag=1, cost_model=TransactionCostModel.from_bps(1, 2, 2))
    assert np.allclose(res.net_returns, res.gross_returns - res.costs)
    expected_cost = res.trades.abs() * (1 + 1 + 2) / 1e4
    assert np.allclose(res.costs, expected_cost)
    assert res.turnover["annualized"] > 0


def test_simulate_rejects_infinite_targets(prices):
    with pytest.raises(QuantProofInputError):
        simulate(prices, pd.Series(np.inf, index=prices.index))


def test_causal_volatility_uses_only_past(prices):
    v1 = causal_volatility(prices["close"])
    changed = prices["close"].copy()
    changed.iloc[300:] *= 1.5
    v2 = causal_volatility(changed)
    assert np.allclose(v1.iloc[:301], v2.iloc[:301], equal_nan=True)


def test_india_provider_known_values():
    p = india_nse_provider()
    fut = p.trade_cost(100_000, "sell", "equity_futures", "2026-05-01")
    assert fut.transaction_tax == pytest.approx(50.0)
    assert fut.total == pytest.approx(50 + 1.73 + 0.1 + 0.18 * (1.73 + 0.1))
    before = p.trade_cost(100_000, "sell", "equity_futures", "2025-05-01")
    assert before.transaction_tax == pytest.approx(20.0)
    deliv = p.trade_cost(100_000, "buy", "equity_delivery", "2025-01-15", brokerage=20)
    assert deliv.total == pytest.approx(100 + 2.97 + 0.1 + 15 + 20 + 0.18 * (20 + 2.97 + 0.1))
    assert p.trade_cost(100_000, "buy", "equity_options", "2026-06-01").transaction_tax == 0.0


def test_provider_coverage_and_errors():
    p = india_nse_provider()
    with pytest.raises(QuantProofInputError, match="Coverage starts 2024-10-01"):
        p.trade_cost(1000, "buy", "equity_delivery", "2023-01-01")
    with pytest.raises(QuantProofInputError, match="Unknown segment"):
        p.schedule_for("crypto", "2025-01-01")
    with pytest.raises(QuantProofInputError):
        p.trade_cost(1000, "short", "equity_delivery", "2025-01-01")
    overlap = CostProvider(
        "x",
        "XX",
        [FeeSchedule("s", dt.date(2020, 1, 1), None), FeeSchedule("s", dt.date(2021, 1, 1), None)],
    )
    with pytest.raises(QuantProofInputError, match="Overlapping"):
        overlap.schedule_for("s", "2022-01-01")
    assert p.describe()["sources"]


def test_provider_cost_component_effective_dated():
    p = india_nse_provider()
    idx = pd.to_datetime(["2026-03-31", "2026-04-01"])
    c = CostContext(
        np.array([-0.1, -0.1]), np.array([100.0, 100.0]), 1_000_000.0, pd.DatetimeIndex(idx)
    )
    costs = ProviderCost(p, "equity_futures").cost(c)
    assert costs[1] > costs[0]  # STT increase effective 2026-04-01
    capped = ProviderCost(p, "equity_delivery", brokerage_bps=3, brokerage_cap=20).cost(
        CostContext(
            np.array([0.1]), np.array([100.0]), 1_000_000.0, pd.DatetimeIndex(["2025-01-02"])
        )
    )
    uncapped = ProviderCost(p, "equity_delivery", brokerage_bps=3).cost(
        CostContext(
            np.array([0.1]), np.array([100.0]), 1_000_000.0, pd.DatetimeIndex(["2025-01-02"])
        )
    )
    assert capped[0] < uncapped[0]


def test_turnover_stats():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    single = pd.Series([1.0, 0.0, -2.0, 1.0], index=idx)
    st = turnover_stats(single, periods_per_year=252)
    assert st["gross_total"] == 4.0 and st["net_total"] == 4.0
    assert st["average_per_bar"] == 1.0 and st["annualized"] == 252.0
    multi = pd.DataFrame({"a": [1.0, -1.0], "b": [-1.0, 0.5]}, index=idx[:2])
    st2 = turnover_stats(multi)
    assert st2["gross_total"] == 3.5 and st2["net_total"] == pytest.approx(0.5)


def test_execution_timing():
    s = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"])
    e = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:59", "2024-01-01 12:05"])
    rep = analyze_execution_timing(s, e)
    assert rep.n_instantaneous == 1 and rep.n_before_signal == 1
    assert rep.min_latency == "-1 days +23:59:00"
    with pytest.raises(QuantProofInputError):
        analyze_execution_timing(s, e[:2])
    with pytest.raises(QuantProofInputError):
        analyze_execution_timing(s.tz_localize("UTC"), e)


def test_roll_spread():
    rng = np.random.default_rng(0)
    true_mid = 100 + np.cumsum(rng.normal(0, 0.01, 5000))
    bounce = true_mid * (1 + 0.001 * rng.choice([-1, 1], 5000))  # 20 bps spread
    est = roll_spread(bounce)
    assert est == pytest.approx(0.002, rel=0.25)
    assert np.isnan(roll_spread([1.0, 2.0]))  # too short to estimate


def test_transaction_tax_sides():
    from quantproof.execution import TransactionTax

    c = ctx([0.5, -0.25, 0.0])
    np.testing.assert_allclose(TransactionTax(10).cost(c), [5e-4, 2.5e-4, 0])
    np.testing.assert_allclose(TransactionTax(10, side="buy").cost(c), [5e-4, 0, 0])
    np.testing.assert_allclose(TransactionTax(10, side="sell").cost(c), [0, 2.5e-4, 0])
    with pytest.raises(QuantProofInputError):
        TransactionTax(10, side="short")


def test_cost_model_compose_without_and_scaled():
    from quantproof.execution import TransactionTax

    model = TransactionCostModel.compose(
        commission=BpsCommission(1),
        spread=FixedSpread(4),
        slippage=None,
        taxes=TransactionTax(3),
    )
    assert model.names == ["commission", "spread", "taxes"]
    c = ctx([1.0, 0.0, -0.5])
    base = model.total(c)
    np.testing.assert_allclose(base, [6e-4, 0, 3e-4])
    np.testing.assert_allclose(model.scaled(2.5).total(c), 2.5 * base)
    np.testing.assert_allclose(model.scaled(0).total(c), 0.0)
    assert model.without("spread", "taxes").names == ["commission"]
    assert model.scaled(2).describe()[0]["multiplier"] == 2.0
    with pytest.raises(QuantProofInputError):
        model.scaled(-1)


def test_execution_section_attribution_and_multipliers(prices):
    from quantproof.analyzers.execution.analyzer import analyze_execution
    from quantproof.config import ExecutionConfig

    signals = np.sign(prices["close"].pct_change(10)).fillna(0.0)
    cfg = ExecutionConfig(tax_bps=1.0, impact_coefficient=0.1)
    section, _findings, sim = analyze_execution(prices, signals, cfg, declared=None)
    rows = section["cost_attribution"]["rows"]
    assert [r["item"] for r in rows] == [
        "gross",
        "commission",
        "spread",
        "slippage",
        "impact",
        "taxes",
        "net",
    ]
    assert sum(r["sum"] for r in rows[:-1]) == pytest.approx(rows[-1]["sum"], abs=1e-12)
    mults = [r["multiplier"] for r in section["cost_sensitivity"]]
    assert mults == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    sharpe = [r["sharpe"] for r in section["cost_sensitivity"]]
    assert all(a >= b for a, b in itertools.pairwise(sharpe))
    zero = section["cost_sensitivity"][0]
    assert zero["sharpe"] == pytest.approx(
        section["scenarios"]["realistic"]["gross_metrics"]["sharpe"]
    )
    one = section["cost_sensitivity"][2]
    assert one["sharpe"] == pytest.approx(section["scenarios"]["realistic"]["metrics"]["sharpe"])
    assert section["break_even_cost_multiplier"] == pytest.approx(
        sim.gross_returns.mean() / sim.costs.mean()
    )
    assert section["semantics"]["audit"].startswith("close-to-close")


@pytest.mark.parametrize("fill", ["intrabar", "event", "VWAP", "twap"])
def test_untestable_declared_fills_warn_instead_of_simulating(prices, fill):
    from quantproof.analyzers.execution.analyzer import analyze_execution
    from quantproof.config import ExecutionConfig

    signals = np.sign(prices["close"].pct_change(10)).fillna(0.0)
    section, findings, _ = analyze_execution(
        prices, signals, ExecutionConfig(), declared={"fill": fill, "signal_lag": 1}
    )
    by_id = {f.id: f for f in findings}
    assert by_id["QP-EXEC-006"].severity.value == "WARN"
    assert "declared" not in section["scenarios"]
    assert section["semantics"]["declared_simulated"] is False


def test_unknown_declared_fill_is_an_input_error(prices):
    from quantproof.analyzers.execution.analyzer import analyze_execution
    from quantproof.config import ExecutionConfig

    with pytest.raises(QuantProofInputError, match="not recognised"):
        analyze_execution(
            prices, prices["close"] * 0, ExecutionConfig(), declared={"fill": "teleport"}
        )


def test_cost_multipliers_must_be_non_negative():
    from pydantic import ValidationError

    from quantproof.config import ExecutionConfig

    with pytest.raises(ValidationError):
        ExecutionConfig(cost_multipliers=[1.0, -0.5])
