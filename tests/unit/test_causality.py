"""Future-perturbation causality test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantproof.analyzers.causal import (
    SCHEMES,
    causality_findings,
    perturb_future,
    run_causality_test,
)
from quantproof.config import CausalityConfig
from quantproof.errors import QuantProofInputError
from quantproof.severity import Severity


def causal(d):
    return np.sign(d["close"].rolling(10).mean() - d["close"].rolling(30).mean())


def causal_ewm(d):
    return d["close"].ewm(span=20).mean() / d["close"] - 1


BROKEN = {
    "negative_shift": lambda d: np.sign(d["close"].shift(-1) - d["close"]),
    "centered": lambda d: (d["close"] > d["close"].rolling(11, center=True).mean()).astype(float),
    "full_sample_zscore": lambda d: (d["close"] - d["close"].mean()) / d["close"].std(),
    "backfill": lambda d: d["close"].where(d.index.dayofweek != 0).bfill().pct_change(),
    "global_rank": lambda d: d["close"].rank(pct=True),
}


@pytest.mark.parametrize("func", [causal, causal_ewm], ids=["rolling", "ewm"])
def test_causal_strategies_pass(prices, func):
    rep = run_causality_test(func, prices)
    assert rep.deterministic and rep.n_fail == 0 and rep.n_pass > 0
    assert rep.control_sensitive is True
    f = causality_findings(rep)
    assert [x.severity for x in f] == [Severity.PASS]


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_non_causal_strategies_fail(prices, name):
    rep = run_causality_test(BROKEN[name], prices)
    assert rep.n_fail > 0, name
    fail = causality_findings(rep, source="x.py")[0]
    assert fail.id == "QP-CAUSAL-001" and fail.severity is Severity.FAIL
    ex = fail.evidence["examples"][0]
    assert ex["first_changed"] <= ex["timestamp"]


def test_perturbation_never_touches_the_past(prices):
    for scheme in SCHEMES:
        for k in (0, 100, len(prices) - 2):
            if scheme == "permutation" and len(prices) - k - 1 < 2:
                continue
            out = perturb_future(prices, k, scheme, seed=1)
            assert out.iloc[: k + 1].equals(prices.iloc[: k + 1].astype(float))
            assert not out.iloc[k + 1 :].equals(prices.iloc[k + 1 :].astype(float))


def test_perturbation_preserves_ohlc_validity(prices):
    for scheme in ("additive", "multiplicative", "shock", "permutation"):
        out = perturb_future(prices, 200, scheme, seed=2)
        assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all()
        assert (out[["open", "high", "low", "close"]] > 0).all().all()


def test_perturbation_errors(prices):
    with pytest.raises(QuantProofInputError):
        perturb_future(prices, 10, "nope")
    with pytest.raises(QuantProofInputError):
        perturb_future(prices, len(prices) - 1, "additive")
    with pytest.raises(QuantProofInputError):
        perturb_future(prices, len(prices) - 2, "permutation")


def test_non_deterministic_strategy_is_inconclusive(prices):
    rng = np.random.default_rng()

    def noisy(d):
        return pd.Series(rng.normal(size=len(d)), index=d.index)

    rep = run_causality_test(noisy, prices)
    assert not rep.deterministic
    f = causality_findings(rep)
    assert f[0].id == "QP-CAUSAL-002" and f[0].severity is Severity.WARN


def test_insensitive_strategy_flagged(prices):
    rep = run_causality_test(lambda d: pd.Series(1.0, index=d.index), prices)
    ids = {f.id for f in causality_findings(rep)}
    assert "QP-CAUSAL-004" in ids


def test_errors_on_perturbed_data_are_reported(prices):
    def fragile(d):
        if d["close"].max() > prices["close"].max() * 1.5:
            raise ValueError("price too high")
        return causal(d)

    rep = run_causality_test(fragile, prices, CausalityConfig(schemes=["shock"]))
    assert rep.n_error > 0
    assert "QP-CAUSAL-003" in {f.id for f in causality_findings(rep)}


def test_deterministic_across_runs(prices):
    a = run_causality_test(BROKEN["full_sample_zscore"], prices, seed=5).to_dict()
    b = run_causality_test(BROKEN["full_sample_zscore"], prices, seed=5).to_dict()
    assert a == b


def test_dataframe_and_array_outputs(prices):
    rep = run_causality_test(lambda d: d[["close"]].rolling(5).mean(), prices)
    assert rep.n_fail == 0
    rep2 = run_causality_test(lambda d: d["close"].rolling(5).mean().to_numpy(), prices)
    assert rep2.n_fail == 0
