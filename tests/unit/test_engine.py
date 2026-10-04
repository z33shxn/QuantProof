"""Audit engine in strategy, static-only and results modes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantproof import AuditConfig, ResearchArtifacts, Severity, audit
from quantproof.errors import QuantProofDataError, QuantProofInputError, QuantProofStrategyError


def causal_strategy(data, fast=10, slow=40):
    c = data["close"]
    return (
        (c.rolling(fast).mean() > c.rolling(slow).mean())
        .astype(float)
        .where(c.rolling(slow).mean().notna())
    )


def test_strategy_callable_full_audit(prices, quick_config):
    res = audit(causal_strategy, prices, config=quick_config)
    assert res.status in (Severity.PASS, Severity.WARN)
    assert "QP-CAUSAL-001" in res.ids(Severity.PASS)
    for key in (
        "overview",
        "data",
        "causality",
        "execution",
        "statistics",
        "validation",
        "regimes",
        "charts",
    ):
        assert key in res.sections, key
    assert res.manifest["content_hash"]
    assert res.sections["overview"]["evaluation_start"] >= prices.index[39].isoformat()
    # verdict is derivable from findings alone
    assert res.summary.failures == sum(f.severity is Severity.FAIL for f in res.findings)


def test_static_only_mode(tmp_path, quick_config):
    p = tmp_path / "s.py"
    p.write_text("def generate_signals(d):\n    return d['close'].shift(-1)\n")
    res = audit(p, None, config=quick_config)
    assert res.status is Severity.FAIL and res.sections["overview"]["mode"] == "static-only"


def test_strategy_errors_are_wrapped(prices, quick_config):
    def broken(data):
        raise KeyError("missing column")

    with pytest.raises(QuantProofStrategyError, match="generate_signals failed"):
        audit(broken, prices, config=quick_config)
    with pytest.raises(QuantProofStrategyError, match="NaN"):
        audit(lambda d: d["close"] * np.nan, prices, config=quick_config)
    with pytest.raises(QuantProofStrategyError, match="infinite"):
        audit(lambda d: d["close"] * np.inf, prices, config=quick_config)
    with pytest.raises(QuantProofInputError, match="Nothing to audit"):
        audit()
    with pytest.raises(QuantProofDataError):
        audit(causal_strategy, prices[["volume"]], config=quick_config)


def test_results_mode_returns_and_trials(quick_config):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020", periods=800, freq="B")
    trials = pd.DataFrame(rng.normal(0, 0.01, (800, 12)), index=idx)
    res = audit(returns=trials.iloc[:, 3], trial_returns=trials, config=quick_config)
    assert res.sections["overview"]["mode"] == "results"
    assert res.sections["statistics"]["dsr"]["n_trials"] == 12
    assert {"QP-VAL-002", "QP-VAL-003", "QP-VAL-004"} <= res.ids()
    assert "QP-CAUSAL-001" in res.ids(Severity.INFO)  # reported as not run


def test_results_mode_trades_features_and_trial_params(prices, quick_config):
    idx = prices.index
    r = prices["close"].pct_change().fillna(0) * 0.5
    trades = pd.DataFrame(
        {"signal_time": idx[10:20], "execution_time": idx[10:20] - pd.Timedelta(hours=1)}
    )
    feats = pd.DataFrame(
        {"good": prices["close"].pct_change(), "bad": prices["close"].pct_change().shift(-1)}
    )
    grid = pd.DataFrame({f"p{i}": r * (1 + 0.1 * i) for i in range(6)})
    params = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 1, 2, 3]})
    art = ResearchArtifacts(
        returns=r,
        prices=prices,
        trades=trades,
        features=feats,
        trial_returns=grid.iloc[1:],
        trial_params=params,
        positions=pd.Series(0.5, index=idx),
        signals=np.sign(feats["good"]).fillna(0),
    )
    res = audit(artifacts=art, config=quick_config)
    assert "QP-EXEC-005" in res.ids(Severity.FAIL)
    assert "QP-LEAK-002" in res.ids(Severity.FAIL)
    assert "sensitivity" in res.sections and "heatmap" in res.sections["sensitivity"]
    assert res.sections["execution"]["turnover"]["gross_total"] == pytest.approx(0.5)


def test_benchmark_enables_trend_regimes(prices, quick_config):
    bench = prices["close"].pct_change().fillna(0)
    res = audit(causal_strategy, prices, config=quick_config, benchmark=bench)
    assert "trend" in res.sections["regimes"]["tables"]
    assert "benchmark_equity" in res.sections["charts"]


def test_trials_override(prices):
    cfg = AuditConfig(quick=True)
    cfg.statistics.trials = 77
    res = audit(causal_strategy, prices, config=cfg)
    assert res.sections["statistics"]["dsr"]["n_trials"] == 77
    assert "QP-STAT-005" not in res.ids()
