"""Adversarial and false-positive suites for the static analyzer.

* ADVERSARIAL: realistic ways look-ahead hides in research code; each must be reported
  with at least the stated severity.
* FALSE_POSITIVES: legitimate code that must NOT produce a FAIL (WARN with context is
  acceptable where the rule is inherently context-dependent, as noted).
* BOUNDARY: patterns outside the documented analysis boundary (Level 3: intra-module data
  flow). They are pinned here so a change in behaviour is a conscious decision, and listed
  in docs/methodology/lookahead-detection.md.
"""

from __future__ import annotations

import textwrap

import pytest

from quantproof.analyzers.static import analyze_source
from quantproof.results import Usage
from quantproof.severity import Severity


def findings(src: str):
    return [
        f
        for f in analyze_source(textwrap.dedent(src), "<adversarial>")
        if f.severity is not Severity.PASS
    ]


def worst(src: str, rule: str) -> Severity | None:
    sev = [f.severity for f in findings(src) if f.id == rule]
    return max(sev, key=lambda s: s.rank) if sev else None


ADVERSARIAL = {
    "period_in_module_constant": (
        "QP001",
        """
        HORIZON = -1
        def generate_signals(df):
            return df['close'].shift(HORIZON)
    """,
    ),
    "future_in_helper": (
        "QP001",
        """
        def future(s):
            return s.shift(-1)
        def generate_signals(df):
            return (future(df['close']) > df['close']).astype(float)
    """,
    ),
    "helper_of_helper": (
        "QP001",
        """
        def lead(s, k):
            return s.shift(-k)
        def tomorrow(s):
            return lead(s, 1)
        def generate_signals(df):
            signal = tomorrow(df['close']) > df['close']
            return signal
    """,
    ),
    "aliased_unbound_method": (
        "QP001",
        """
        import pandas as pd
        nxt = pd.Series.shift
        def generate_signals(df):
            return nxt(df['close'], -1) > df['close']
    """,
    ),
    "aliased_import": (
        "QP004",
        """
        from sklearn.model_selection import train_test_split as tts
        a, b = tts(X)
    """,
    ),
    "centered_hidden_in_helper": (
        "QP002",
        """
        def smooth(s):
            return s.rolling(21, center=True).mean()
        def generate_signals(df):
            return (df['close'] > smooth(df['close'])).astype(float)
    """,
    ),
    "helper_mutates_dataframe": (
        "QP001",
        """
        def add_features(df):
            df['fwd'] = df['close'].pct_change().shift(-1)
        def generate_signals(df):
            add_features(df)
            position = df['fwd'] > 0
            return position
    """,
    ),
    "future_merged_frame": (
        "QP001",
        """
        def generate_signals(df):
            nxt = df[['close']].shift(-1).rename(columns={'close': 'nxt'})
            merged = df.join(nxt)
            return merged['nxt'] > merged['close']
    """,
    ),
    "lambda_pipe": (
        "QP001",
        """
        def generate_signals(df):
            return df['close'].pipe(lambda s: s.shift(-1) > s)
    """,
    ),
    "comprehension": (
        "QP001",
        """
        def generate_signals(df):
            leads = [df[c].shift(-1) for c in ['close']]
            return leads[0] > df['close']
    """,
    ),
    "method_in_class": (
        "QP001",
        """
        class Strategy:
            def _peek(self, s):
                return s.shift(-1)
            def signals(self, df):
                return self._peek(df['close']) > df['close']
    """,
    ),
    "chained_pandas": (
        "QP001",
        """
        def generate_signals(df):
            return df['close'].pct_change().shift(-1).rolling(5).mean() > 0
    """,
    ),
    "backfill_in_signal": (
        "QP015",
        """
        def generate_signals(df):
            px = df['close'].bfill()
            return (px > px.rolling(10).mean()).astype(float)
    """,
    ),
    "global_scaler_then_split": (
        "QP006",
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        Xs = StandardScaler().fit_transform(X)
        for tr, te in TimeSeriesSplit(5).split(Xs):
            pass
    """,
    ),
    "target_renamed": (
        "QP008",
        """
        y = df['target']
        df['alpha_signal_v2'] = df['target']
        X = df[['ret_1', 'alpha_signal_v2']]
        model.fit(X, y)
    """,
    ),
    "future_return_feature": (
        "QP012",
        """
        df['fwd'] = df['close'].pct_change(-5)
        X = df[['ret_1', 'fwd']]
        model.fit(X, df['y'])
    """,
    ),
    "same_bar_close": (
        "QP010",
        """
        df['ret'] = df['close'].pct_change()
        df['signal'] = (df['close'] > df['close'].rolling(20).mean()).astype(float)
        df['fee'] = 0.0001
        df['pnl'] = df['signal'] * df['ret'] - df['fee']
    """,
    ),
    "huge_search": (
        "QP013",
        """
        import itertools
        for a, b, c in itertools.product(range(20), range(20), range(20)):
            pass
    """,
    ),
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_cases_are_caught(name):
    rule, src = ADVERSARIAL[name]
    sev = worst(src, rule)
    assert sev is not None, f"{rule} missed in {name}"
    if rule in {"QP001", "QP002", "QP006", "QP008", "QP010", "QP012", "QP015"}:
        assert sev is Severity.FAIL, f"{name}: {rule} only {sev}"


def test_live_decision_usage_is_labelled():
    f = next(f for f in findings(ADVERSARIAL["future_in_helper"][1]) if f.id == "QP001")
    assert f.usage is Usage.LIVE_DECISION
    assert "live decision" in f.message


FALSE_POSITIVES = {
    # Centered smoothing for a descriptive chart: WARN with context, never FAIL.
    "centered_for_plot": """
        import matplotlib.pyplot as plt
        def plot(df):
            trend = df['close'].rolling(21, center=True).mean()
            plt.plot(trend)
    """,
    # Supervised label built from future returns, never used as a feature.
    "label_construction": """
        def research(df, model):
            df['target'] = (df['close'].pct_change().shift(-5) > 0).astype(int)
            X = df[['ret_1', 'vol_20']]
            model.fit(X, df['target'])
    """,
    # Normalization fitted after a temporal split, on the training part only.
    "normalize_after_split": """
        from sklearn.preprocessing import StandardScaler
        train, test = df.iloc[:500], df.iloc[500:]
        scaler = StandardScaler().fit(train[cols])
        Xtr, Xte = scaler.transform(train[cols]), scaler.transform(test[cols])
    """,
    # Decision made at the open with the prior close; earns open→close of the same bar.
    "open_decision_same_bar": """
        def backtest(df):
            signal = (df['open'] > df['close'].shift(1)).astype(float)
            intraday = df['close'] / df['open'] - 1
            cost_bps = 1
            return signal * intraday - signal.diff().abs() * cost_bps / 1e4
    """,
    # Cross-sectional z-score within each date.
    "cross_sectional_zscore": """
        z = panel.groupby('date')['mom'].transform(lambda s: (s - s.mean()) / s.std())
    """,
    # Lagged target as a feature (past information).
    "lagged_target_feature": """
        y = df['target']
        df['target_lag'] = df['target'].shift(1)
        model.fit(df[['ret_1', 'target_lag']], y)
    """,
    # Random split on i.i.d. synthetic data: WARN (context-dependent), not FAIL.
    "iid_random_split": """
        from sklearn.model_selection import train_test_split
        X_tr, X_te = train_test_split(X_iid, random_state=0)
    """,
    # Rolling, expanding and EWM features; positive shifts.
    "causal_features": """
        def generate_signals(df):
            c = df['close']
            z = (c - c.rolling(20).mean()) / c.rolling(20).std()
            trend = c.ewm(span=50).mean() > c.expanding().mean()
            return ((z < 1) & trend & (c > c.shift(1))).astype(float)
    """,
}


@pytest.mark.parametrize("name", sorted(FALSE_POSITIVES))
def test_legitimate_code_is_not_failed(name):
    fails = [f for f in findings(FALSE_POSITIVES[name]) if f.severity is Severity.FAIL]
    assert fails == [], [(f.id, f.message) for f in fails]


def test_label_construction_is_marked_legitimate():
    f = next(f for f in findings(FALSE_POSITIVES["label_construction"]) if f.id == "QP001")
    assert f.severity is Severity.INFO and f.usage is Usage.LABEL_OR_ANALYSIS


def test_plot_only_centered_window_is_warn_with_context():
    f = next(f for f in findings(FALSE_POSITIVES["centered_for_plot"]) if f.id == "QP002")
    assert f.severity is Severity.WARN and "analysis" in f.message


BOUNDARY = {
    # Containers filled through method calls (append/update) are not traced.
    "container_mutation": """
        def generate_signals(df):
            buf = []
            buf.append(df['close'].shift(-1))
            signal = buf[0] > df['close']
            return signal
    """,
    # Dynamic attribute access is not resolved.
    "getattr_dynamic": """
        def generate_signals(df):
            fn = getattr(df['close'], 'sh' + 'ift')
            return fn(-1) > df['close']
    """,
}


def test_containers_built_by_literals_are_traced():
    src = """
        def generate_signals(df):
            store = {'x': df['close'].shift(-1)}
            signal = store['x'] > df['close']
            return signal
    """
    assert worst(src, "QP001") is Severity.FAIL


def test_documented_boundary_container_mutation():
    # The negative shift is still reported, but cannot be traced through append(): WARN.
    assert worst(BOUNDARY["container_mutation"], "QP001") is Severity.WARN


def test_documented_boundary_dynamic_attribute():
    assert worst(BOUNDARY["getattr_dynamic"], "QP001") is None
