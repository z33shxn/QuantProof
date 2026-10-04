"""Strategy contract/loader and adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantproof import PandasAdapter, ResearchArtifacts, load_strategy
from quantproof.errors import QuantProofInputError, QuantProofStrategyError
from quantproof.strategy import normalize_signals


def write(tmp_path, body, name="s.py"):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_load_file_with_metadata(tmp_path):
    p = write(
        tmp_path,
        "NAME='demo'\nPARAMETERS={'n': 3}\nPARAM_GRID={'n': [2, 3, 4], 'm': [1, 2]}\n"
        "EXECUTION={'signal_lag': 1, 'commission_bps': 1.0}\n"
        "def generate_signals(data, n=3, m=1):\n    return data['close'].rolling(n).mean() * 0 + m\n",
    )
    spec = load_strategy(p)
    assert spec.name == "demo" and spec.parameters == {"n": 3} and spec.grid_size == 6
    assert spec.execution == {"signal_lag": 1, "commission_bps": 1.0}
    assert len(spec.grid()) == 6 and spec.grid()[0] == {"m": 1, "n": 2}
    sub = spec.grid(max_trials=4, seed=1)
    assert len(sub) == 4 and sub == spec.grid(max_trials=4, seed=1)
    idx = pd.date_range("2024", periods=5, freq="D")
    out = spec(pd.DataFrame({"close": np.arange(5.0)}, index=idx), m=2)
    assert out.iloc[-1] == 2.0


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("x = 1\n", "does not define"),
        ("raise RuntimeError('boom')\n", "Importing"),
        (
            "EXECUTION = {'lag': 1}\ndef generate_signals(d):\n    return d\n",
            "unknown EXECUTION keys",
        ),
        ("PARAM_GRID = {'a': []}\ndef generate_signals(d):\n    return d\n", "non-empty list"),
        ("PARAMETERS = [1]\ndef generate_signals(d):\n    return d\n", "must be dicts"),
    ],
)
def test_bad_strategy_files(tmp_path, body, match):
    with pytest.raises(QuantProofStrategyError, match=match):
        load_strategy(write(tmp_path, body))


def test_load_errors_for_paths(tmp_path):
    with pytest.raises(QuantProofStrategyError, match="not found"):
        load_strategy(tmp_path / "missing.py")
    with pytest.raises(QuantProofStrategyError, match=r"\.py"):
        load_strategy(write(tmp_path, "x", "s.txt"))


def test_callable_and_module_sources():
    def generate_signals(data):
        return data["close"] * 0

    spec = load_strategy(generate_signals)
    assert spec.name == "generate_signals" and load_strategy(spec) is spec
    import types

    mod = types.ModuleType("m")
    mod.generate_signals = generate_signals  # type: ignore[attr-defined]
    assert load_strategy(mod).func is generate_signals


def test_normalize_signals():
    idx = pd.date_range("2024", periods=3, freq="D")
    assert normalize_signals(np.array([1, 0, 1]), idx).tolist() == [1.0, 0.0, 1.0]
    sub = normalize_signals(pd.Series([1.0], index=idx[1:2]), idx)
    assert np.isnan(sub.iloc[0]) and sub.iloc[1] == 1.0
    with pytest.raises(QuantProofStrategyError):
        normalize_signals(np.zeros(2), idx)
    with pytest.raises(QuantProofStrategyError):
        normalize_signals(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}, index=idx), idx)
    with pytest.raises(QuantProofStrategyError):
        normalize_signals(pd.Series([1.0], index=[pd.Timestamp("1999-01-01")]), idx)


def test_pandas_adapter(tmp_path):
    idx = pd.date_range("2024", periods=50, freq="B")
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50), index=idx)
    trials = pd.DataFrame({"a": r, "b": -r})
    art = PandasAdapter().load(
        {"returns": r, "trial_returns": trials, "trial_params": pd.DataFrame({"x": [1, 2]})},
        metadata={"name": "t"},
    )
    assert art.metadata["adapter"] == "pandas" and art.metadata["name"] == "t"
    with pytest.raises(QuantProofInputError, match="Unknown artifact"):
        PandasAdapter().load({"pnl": r})
    with pytest.raises(QuantProofInputError, match="trial_params"):
        PandasAdapter().load({"trial_returns": trials, "trial_params": pd.DataFrame({"x": [1]})})
    pd.DataFrame({"date": idx, "returns": r.to_numpy()}).to_csv(tmp_path / "r.csv", index=False)
    assert len(PandasAdapter().load(returns=tmp_path / "r.csv").returns) == 50
    with pytest.raises(QuantProofInputError, match="empty"):
        ResearchArtifacts().validate()
    art2 = ResearchArtifacts(returns=r, trades=pd.DataFrame({"qty": [1]}))
    art2.validate()
    assert "execution-timing checks skipped" in art2.metadata["notes"][0]
