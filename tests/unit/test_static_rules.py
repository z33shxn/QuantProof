"""Static analyzer: every rule is triggered by a deliberately broken fixture and stays
silent on correct code. Deliberately broken files live in tests/fixtures/broken."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from quantproof.analyzers.static import (
    RULES,
    analyze_file,
    analyze_path,
    analyze_source,
    list_rules,
)
from quantproof.config import StaticConfig
from quantproof.severity import Severity

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

EXPECTED = {
    "qp001_negative_shift.py": ("QP001", Severity.FAIL),
    "qp002_centered_rolling.py": ("QP002", Severity.FAIL),
    "qp003_lookahead_index.py": ("QP003", Severity.FAIL),
    "qp004_random_split.py": ("QP004", Severity.WARN),
    "qp005_shuffled_kfold.py": ("QP005", Severity.WARN),
    "qp006_global_scaler.py": ("QP006", Severity.FAIL),
    "qp007_full_sample_zscore.py": ("QP007", Severity.WARN),
    "qp008_target_in_features.py": ("QP008", Severity.FAIL),
    "qp009_same_bar.py": ("QP009", Severity.WARN),
    "qp010_missing_lag.py": ("QP010", Severity.FAIL),
    "qp011_no_costs.py": ("QP011", Severity.WARN),
    "qp012_future_feature.py": ("QP012", Severity.FAIL),
    "qp013_large_grid.py": ("QP013", Severity.WARN),
    "qp014_no_oos.py": ("QP014", Severity.WARN),
    "qp015_backfill.py": ("QP015", Severity.FAIL),
}


def issues(source: str, **cfg: object) -> dict[str, list[Severity]]:
    out: dict[str, list[Severity]] = {}
    for f in analyze_source(textwrap.dedent(source), "<test>", StaticConfig(**cfg)):
        if f.severity is not Severity.PASS:
            out.setdefault(f.id, []).append(f.severity)
    return out


def test_every_rule_has_a_broken_fixture():
    assert {r.id for r in RULES} == {rule for rule, _ in EXPECTED.values()}
    assert len(list_rules()) == 15


@pytest.mark.parametrize(("name", "expected"), sorted(EXPECTED.items()))
def test_broken_fixture_is_caught(name, expected):
    rule, severity = expected
    findings = analyze_file(FIXTURES / "broken" / name)
    hits = [f for f in findings if f.id == rule and f.severity is severity]
    assert hits, f"{rule} not reported as {severity} for {name}"
    f = hits[0]
    assert f.location is not None and f.location.line is not None and f.location.file
    assert f.why_it_matters and f.recommendation and f.confidence is not None


@pytest.mark.parametrize("path", sorted((FIXTURES / "clean").glob("*.py")), ids=lambda p: p.name)
def test_clean_fixtures_have_no_issues(path):
    bad = [f for f in analyze_file(path) if f.is_issue]
    assert bad == [], [(f.id, f.message) for f in bad]


def test_clean_example_strategy_has_no_issues():
    root = Path(__file__).resolve().parents[2]
    bad = [f for f in analyze_file(root / "examples/clean_strategy/strategy.py") if f.is_issue]
    assert bad == []


def test_passes_are_reported_for_every_rule():
    findings = analyze_source("x = 1\n")
    assert {f.id for f in findings} == {r.id for r in RULES}
    assert all(f.severity is Severity.PASS for f in findings)


# ------------------------------------------------------------------- negatives
@pytest.mark.parametrize(
    "src",
    [
        "s = df['close'].shift(1)",
        "s = df['close'].shift(periods=3)",
        "s = df['close'].diff()",
        "m = df['close'].rolling(20, center=False).mean()",
        "from sklearn.model_selection import train_test_split\na, b = train_test_split(X, shuffle=False)",
        "from sklearn.model_selection import TimeSeriesSplit\ncv = TimeSeriesSplit(5)",
        "z = (x - x.rolling(20).mean()) / x.rolling(20).std()",
        "z = df.groupby('date')['x'].transform(lambda s: (s - s.mean()) / s.std())",
        "z = (X_test - X_train.mean()) / X_train.std()",
        "y = df['close'].ffill()",
    ],
)
def test_valid_patterns_are_not_flagged(src):
    assert not {k for k, v in issues(src).items() if Severity.WARN in v or Severity.FAIL in v}


def test_negative_shift_label_only_is_info():
    src = """
    def research(df):
        df["label"] = df["close"].pct_change().shift(-1)
        X = df[["ret_1"]]
        model.fit(X, df["label"])
    """
    assert issues(src)["QP001"] == [Severity.INFO]


def test_negative_shift_unclassified_is_warn():
    assert issues("tmp = df['close'].shift(-3)\n")["QP001"] == [Severity.WARN]


def test_negative_shift_via_variable_and_kwarg():
    src = """
    def generate_signals(df, h=1):
        nxt = df['close'].shift(periods=-h)
        return (nxt > df['close']).astype(float)
    """
    assert Severity.FAIL in issues(src)["QP001"]


def test_taint_flows_through_columns_and_assign():
    src = """
    def generate_signals(df):
        df = df.assign(lead=df['close'].pct_change(-1))
        position = df['lead'] > 0
        return position
    """
    assert Severity.FAIL in issues(src)["QP001"]


def test_reassignment_clears_taint():
    src = """
    def generate_signals(df):
        x = df['close'].shift(-1)
        x = df['close'].shift(1)
        signal = x > 0
        return signal
    """
    assert issues(src).get("QP001") == [Severity.WARN]


def test_np_roll_is_flagged():
    src = "import numpy as np\nprev = np.roll(arr, 1)\n"
    assert issues(src)["QP003"] == [Severity.WARN]


def test_centered_rolling_non_literal_is_low_confidence_warn():
    assert issues("m = s.rolling(5, center=flag).mean()")["QP002"] == [Severity.WARN]


def test_kfold_without_shuffle_is_info_and_search_default_cv_info():
    src = "from sklearn.model_selection import KFold, GridSearchCV\nk = KFold(5)\ng = GridSearchCV(m, {'a': [1]}, cv=5)\n"
    found = issues(src)
    assert found["QP005"] == [Severity.INFO, Severity.INFO]


def test_fit_after_split_on_train_is_fine():
    src = """
    from sklearn.preprocessing import StandardScaler
    X_train, X_test = X[:100], X[100:]
    scaler = StandardScaler().fit(X_train)
    """
    assert "QP006" not in issues(src)


def test_transformer_without_any_split_is_qp007():
    src = "from sklearn.decomposition import PCA\np = PCA(3)\nZ = p.fit_transform(X)\n"
    assert issues(src)["QP007"] == [Severity.WARN]


def test_target_not_dropped_from_features():
    src = """
    y = df["target"]
    X = df.drop(columns=["date"])
    model.fit(X, y)
    """
    assert issues(src)["QP008"] == [Severity.FAIL]
    ok = issues("y = df['target']\nX = df.drop(columns=['target'])\nmodel.fit(X, y)\n")
    assert "QP008" not in ok


def test_same_expression_as_features_and_target():
    assert issues("model.fit(X, X)\n")["QP008"] == [Severity.FAIL]


def test_declared_execution_lag_zero_and_zero_costs():
    src = 'EXECUTION = {"signal_lag": 0, "commission_bps": 0}\n'
    found = issues(src)
    assert found["QP009"] == [Severity.WARN] and found["QP011"] == [Severity.WARN]


def test_qp010_skips_aligned_alternatives():
    assert "QP010" not in issues("pnl = signal * returns.shift(-1)\n")
    assert "QP010" not in issues("pnl = position.shift(2) * returns\n")
    # Provenance unknown: execution semantics cannot be determined → WARN, not FAIL.
    assert issues("pnl = df['pos'].mul(df['ret'])\n")["QP010"] == [Severity.WARN]
    traced = "df['ret'] = df['close'].pct_change()\ndf['pos'] = df['close'] > 1\npnl = df['pos'].mul(df['ret'])\n"
    assert issues(traced)["QP010"] == [Severity.FAIL]


def test_qp013_threshold_is_configurable():
    src = "for a in range(10):\n    for b in range(5):\n        pass\n"
    assert issues(src)["QP013"] == [Severity.INFO]
    assert issues(src, max_trials_threshold=20)["QP013"] == [Severity.WARN]
    assert issues("PARAM_GRID = {'a': [1, 2, 3], 'b': [1, 2]}\n")["QP013"] == [Severity.INFO]


def test_param_grid_does_not_trigger_qp014():
    """Regression: a declared PARAM_GRID is evaluated out of sample by QuantProof."""
    src = "PARAM_GRID = {'fast': [5, 10], 'slow': [50, 100]}\ndef generate_signals(d, fast, slow):\n    return d['close']\n"
    assert "QP014" not in issues(src)


def test_qp015_variants():
    found = issues(
        "from scipy.signal import filtfilt, savgol_filter\n"
        "a = filtfilt(b, a, x)\nc = savgol_filter(x, 11, 2)\n"
        "d = s.fillna(method='bfill')\ne = s.interpolate(method='ffill')\n"
        "import numpy as np\nf = np.convolve(x, k, mode='same')\ng = np.fft.fft(x)\n"
    )
    assert len(found["QP015"]) == 5  # ffill interpolation is causal


def test_inline_suppression_turns_issue_into_info():
    src = "s = df['close'].rolling(5, center=True).mean()  # quantproof: ignore[QP002]\n"
    found = issues(src)
    assert found["QP002"] == [Severity.INFO]
    src_all = "s = df['close'].rolling(5, center=True).mean()  # quantproof: ignore\n"
    assert issues(src_all)["QP002"] == [Severity.INFO]
    other = "s = df['close'].rolling(5, center=True).mean()  # quantproof: ignore[QP001]\n"
    assert issues(other)["QP002"] == [Severity.WARN]  # not traced to a decision: WARN


def test_syntax_error_is_reported_not_raised():
    findings = analyze_source("def broken(:\n", "bad.py")
    assert findings[0].id == "QP000" and findings[0].severity is Severity.FAIL


def test_disabled_rules_and_directory_scan(tmp_path):
    (tmp_path / "a.py").write_text("x = s.shift(-1)\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("m = s.rolling(3, center=True).mean()\n")
    found = {f.id for f in analyze_path(tmp_path)}
    assert {"QP001", "QP002"} <= found
    quiet = analyze_source("x = s.shift(-1)\n", config=StaticConfig(disabled_rules=["QP001"]))
    assert "QP001" not in {f.id for f in quiet}


def test_import_alias_resolution():
    src = "from sklearn.model_selection import train_test_split as tts\nX_tr, X_te = tts(X)\n"
    assert issues(src)["QP004"] == [Severity.WARN]
    src2 = "import sklearn.model_selection as ms\na = ms.ShuffleSplit(5)\n"
    assert issues(src2)["QP005"] == [Severity.WARN]
