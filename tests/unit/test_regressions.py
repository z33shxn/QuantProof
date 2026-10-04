"""One test per bug found while building QuantProof (see CHANGELOG)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantproof.analyzers.execution.analyzer import analyze_execution
from quantproof.analyzers.static import analyze_source
from quantproof.analyzers.statistical.selection import analyze_selection
from quantproof.config import ExecutionConfig, StatisticsConfig, ValidationConfig
from quantproof.data import load_frame, validate_data
from quantproof.execution import TransactionCostModel, simulate
from quantproof.execution.slippage import SizeSlippage
from quantproof.severity import Severity
from quantproof.validation import PurgedKFold
from quantproof.validation.temporal import label_intervals

SRC = Path(__file__).resolve().parents[2] / "src" / "quantproof"


def test_gap_detection_uses_true_nanoseconds_for_microsecond_indexes(tmp_path):
    """pandas may parse timestamps at microsecond resolution; asi8 is then not ns."""
    idx = pd.date_range("2024-01-01", periods=30, freq="B").as_unit("us")
    df = pd.DataFrame({"close": np.linspace(1, 2, 30)}, index=idx)
    f = next(x for x in validate_data(df) if x.id == "QP-DATA-009")
    assert f.evidence["median_spacing"] == "1 days 00:00:00"
    assert f.evidence["session_breaks_ignored"] == 0


def test_label_intervals_resolution_independent():
    idx = pd.date_range("2024", periods=5, freq="D")
    t1 = pd.Series(idx + pd.Timedelta(days=1), index=idx)
    t1_us = pd.Series((idx + pd.Timedelta(days=1)).as_unit("us"), index=idx.as_unit("s"))
    a = label_intervals(5, t1)
    b = label_intervals(5, t1_us)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_embargo_starts_after_test_label_span():
    """Embargo is additive to purging (López de Prado 2018, snippet 7.3)."""
    for train, test in PurgedKFold(4, label_horizon=3, embargo=2).split(np.zeros(40)):
        after = train[train > test.max()]
        if after.size:
            assert after.min() == test.max() + 3 + 2 + 1


def test_param_grid_alone_is_not_an_unvalidated_search():
    src = "PARAM_GRID = {'a': [1, 2]}\ndef generate_signals(d, a=1):\n    return d['close']\n"
    assert not [f for f in analyze_source(src) if f.id == "QP014" and f.is_issue]


def test_walk_forward_warns_when_oos_negative_even_if_is_negative():
    rng = np.random.default_rng(0)
    m = pd.DataFrame(
        rng.normal(-0.001, 0.01, (600, 4)), index=pd.date_range("2020", periods=600, freq="B")
    )
    cfg = ValidationConfig(pbo=False, cpcv=False, reality_check=False)
    _, findings = analyze_selection(m, cfg, StatisticsConfig())
    f = next(x for x in findings if x.id == "QP-VAL-001")
    assert f.severity is Severity.WARN


def test_future_feature_reported_once_per_origin():
    src = (
        "def f(df, model):\n"
        "    df['fwd'] = df['close'].shift(-1)\n"
        "    X = df[['fwd']]\n"
        "    Xs = scaler.fit_transform(X)\n"
        "    model.fit(Xs, y)\n"
        "    model.predict(Xs)\n"
    )
    hits = [f for f in analyze_source(src) if f.id == "QP012"]
    assert len(hits) == 1 and hits[0].evidence["sink_lines"] == [4, 5, 6]


def test_import_alias_resolves_to_real_name():
    src = "from sklearn.model_selection import train_test_split as tts\na, b = tts(X)\n"
    assert any(f.id == "QP004" and f.severity is Severity.WARN for f in analyze_source(src))


def test_zero_cost_declaration_with_lagged_fills_can_fail(prices):
    """EXEC-001 must consider zero declared costs, not only same-bar fills."""
    r = prices["close"].pct_change()
    # Constructed so that, filled with one bar of lag, the position earns |r| every bar:
    # a large gross edge with ~1x turnover per bar that realistic costs wipe out.
    sig = np.sign(r.shift(-2)).fillna(0)
    declared = {"signal_lag": 1, "commission_bps": 0.0}
    cfg = ExecutionConfig(commission_bps=60, spread_bps=60, slippage_bps=60)
    section, findings, _ = analyze_execution(prices, sig, cfg, declared=declared)
    f = next(x for x in findings if x.id == "QP-EXEC-001")
    assert section["scenarios"]["declared"]["metrics"]["sharpe"] > 5
    assert f.severity is Severity.FAIL
    assert "zero transaction costs" in f.message


def test_size_costs_do_not_use_future_volume(prices):
    """ADV for the first bar must not be back-filled from later bars."""
    sig = pd.Series(1.0, index=prices.index)
    a = simulate(prices, sig, lag=0, cost_model=TransactionCostModel([SizeSlippage(10)]))
    p2 = prices.copy()
    p2.iloc[1:, p2.columns.get_loc("volume")] = 1
    b = simulate(p2, sig, lag=0, cost_model=TransactionCostModel([SizeSlippage(10)]))
    assert a.costs.iloc[0] == b.costs.iloc[0]


def test_sources_parse_under_python_310_grammar():
    """f-strings with backslashes in expressions are a SyntaxError before Python 3.12."""
    if sys.version_info >= (3, 12):
        for path in SRC.rglob("*.py"):
            tree = ast.parse(path.read_text(), feature_version=(3, 10))
            assert tree is not None
    else:  # pragma: no cover - exercised by the 3.10/3.11 CI jobs
        pytest.skip("native interpreter already enforces the grammar")


def test_loader_handles_microsecond_csv_dates():
    df = load_frame(Path(__file__).resolve().parents[2] / "examples/data/prices.csv")
    f = next(x for x in validate_data(df) if x.id == "QP-DATA-009")
    assert f.severity is Severity.PASS and f.evidence["median_spacing"] == "1 days 00:00:00"
