"""Regime labelling (causal, documented) and parameter-surface analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantproof.config import RegimeConfig
from quantproof.errors import QuantProofInputError
from quantproof.regimes import (
    drawdown_regimes,
    drawdown_series,
    performance_by_regime,
    regime_analysis,
    trend_regimes,
    volatility_regimes,
)
from quantproof.sensitivity import (
    analyze_parameter_surface,
    render_ascii_surface,
    sensitivity_findings,
    surface_matrix,
)
from quantproof.severity import Severity


def test_drawdown_series_and_labels():
    r = pd.Series([0.0, 0.10, -0.05, -0.10, -0.10, 0.30])
    dd = drawdown_series(r)
    assert dd.iloc[2] == pytest.approx(-0.05)
    assert dd.iloc[4] == pytest.approx(1.1 * 0.95 * 0.9 * 0.9 / 1.1 - 1)
    lab = drawdown_regimes(r, thresholds=[-0.10, -0.20])
    assert pd.isna(lab.iloc[0])
    # label at t uses drawdown at t-1
    assert lab.iloc[3] == "normal" and lab.iloc[4] == "drawdown" and lab.iloc[5] == "deep_drawdown"


def test_regime_labels_are_causal():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 400), index=pd.date_range("2020", periods=400, freq="B"))
    v1 = volatility_regimes(r, window=20, thresholds=[0.1, 0.2])
    t1 = trend_regimes(r, window=30)
    d1 = drawdown_regimes(r)
    r2 = r.copy()
    r2.iloc[200] = 0.5  # shock at t=200 must not affect labels at t<=200
    assert v1.iloc[:201].equals(volatility_regimes(r2, window=20, thresholds=[0.1, 0.2]).iloc[:201])
    assert t1.iloc[:201].equals(trend_regimes(r2, window=30).iloc[:201])
    assert d1.iloc[:201].equals(drawdown_regimes(r2).iloc[:201])


def test_volatility_terciles():
    rng = np.random.default_rng(1)
    r = pd.Series(np.r_[rng.normal(0, 0.005, 300), rng.normal(0, 0.03, 300)])
    lab = volatility_regimes(r, window=20)
    counts = lab.value_counts()
    assert set(counts.index) == {"low", "normal", "high"}
    assert lab.iloc[-50:].eq("high").mean() > 0.9
    with pytest.raises(QuantProofInputError):
        volatility_regimes(r, thresholds=[0.1], labels=["a", "b", "c"])


def test_performance_by_regime_and_min_obs():
    r = pd.Series([0.01, -0.01, 0.02, 0.0, 0.01, 0.03])
    lab = pd.Series(["a", "a", "a", "b", "b", "b"])
    rows = performance_by_regime(r, lab, periods_per_year=1, min_observations=3)
    assert rows[0]["regime"] == "a" and rows[0]["n_observations"] == 3
    assert rows[0]["sharpe"] == pytest.approx(
        np.mean([0.01, -0.01, 0.02]) / np.std([0.01, -0.01, 0.02], ddof=1)
    )
    few = performance_by_regime(r, lab, min_observations=4)
    assert few[0]["sharpe"] is None and not few[0]["sufficient_data"]


def test_regime_analysis_flags_negative_regime():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2018", periods=900, freq="B")
    ref = pd.Series(np.r_[rng.normal(0, 0.005, 450), rng.normal(0, 0.03, 450)], index=idx)
    strat = pd.Series(
        np.r_[rng.normal(0.003, 0.005, 450), rng.normal(-0.001, 0.005, 450)], index=idx
    )
    section, findings = regime_analysis(strat, ref, RegimeConfig(), benchmark_supplied=True)
    assert set(section["tables"]) == {"volatility", "drawdown", "trend"}
    assert findings[0].id == "QP-REGIME-001" and findings[0].severity is Severity.WARN
    assert "trend" in section["definitions"]


def test_regime_config_validation():
    with pytest.raises(ValueError, match="increasing"):
        RegimeConfig(vol_quantiles=[0.6, 0.3])
    with pytest.raises(ValueError, match="one more entry"):
        RegimeConfig(vol_labels=["a", "b"])
    with pytest.raises(ValueError, match="decreasing"):
        RegimeConfig(drawdown_thresholds=[-0.2, -0.1])


def grid(fn):
    return pd.DataFrame(
        [{"a": a, "b": b, "sharpe": fn(a, b)} for a in range(1, 6) for b in range(1, 6)]
    )


def test_robust_and_fragile_surfaces():
    robust = analyze_parameter_surface(
        grid(lambda a, b: 1.0 - 0.02 * (abs(a - 3) + abs(b - 3))), ["a", "b"]
    )
    assert robust.classification == "robust" and robust.best == {"a": 3, "b": 3}
    assert robust.n_neighbours == 8
    spike = analyze_parameter_surface(
        grid(lambda a, b: 3.0 if (a, b) == (2, 4) else 0.1), ["a", "b"]
    )
    assert spike.classification == "fragile" and spike.spike_z > 3
    moderate = analyze_parameter_surface(
        grid(lambda a, b: 1.0 if (a, b) == (3, 3) else 0.55), ["a", "b"]
    )
    assert moderate.classification == "moderate"
    negative = analyze_parameter_surface(grid(lambda a, b: -1.0 - a), ["a", "b"])
    assert negative.classification == "undetermined"
    sev = {f.id: f.severity for f in sensitivity_findings(spike)}
    assert sev["QP-SENS-001"] is Severity.WARN
    assert sensitivity_findings(robust)[0].severity is Severity.PASS


def test_local_maxima_and_corner_neighbours():
    surf = grid(lambda a, b: float((a, b) in {(1, 1), (5, 5)}))
    res = analyze_parameter_surface(surf, ["a", "b"])
    assert res.n_neighbours == 3  # corner point
    assert len(res.local_maxima) >= 2


def test_is_oos_spearman():
    df = grid(lambda a, b: a + b)
    df["oos"] = -(df["a"] + df["b"])
    res = analyze_parameter_surface(df, ["a", "b"], oos_metric="oos")
    assert res.is_oos_spearman == pytest.approx(-1.0)
    assert res.oos_best_matches_is_best is False
    f = next(x for x in sensitivity_findings(res) if x.id == "QP-SENS-002")
    assert f.severity is Severity.WARN


def test_surface_matrix_and_ascii():
    df = grid(lambda a, b: a * 10 + b)
    mat = surface_matrix(df, "a", "b", "sharpe")
    assert mat.shape == (5, 5) and mat.loc[2, 3] == 32
    txt = render_ascii_surface(df, "a", "b", "sharpe")
    assert "55.00★" in txt and txt.count("\n") == 5


def test_surface_errors():
    with pytest.raises(QuantProofInputError):
        analyze_parameter_surface(pd.DataFrame({"a": [1]}), ["a"], "sharpe")
    with pytest.raises(QuantProofInputError):
        analyze_parameter_surface(pd.DataFrame({"a": [1], "sharpe": [1.0]}), ["a"])
