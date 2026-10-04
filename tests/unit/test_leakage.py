"""Runtime leakage diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantproof.audit.leakage import feature_leakage_findings, signal_foresight_finding
from quantproof.audit.severity import Severity


def test_feature_copies_target_and_future_returns(prices):
    r = prices["close"].pct_change()
    target = (r.shift(-1) > 0).astype(float)
    feats = pd.DataFrame(
        {
            "lag": r.shift(1),
            "copy": target * 2 + 1,
            "future3": r.shift(-3),
            "noise": np.random.default_rng(0).normal(size=len(r)),
        },
        index=prices.index,
    )
    fs = {f.id: f for f in feature_leakage_findings(feats, target=target, asset_returns=r)}
    assert fs["QP-LEAK-001"].severity is Severity.FAIL
    assert [d["feature"] for d in fs["QP-LEAK-001"].evidence["features"]] == ["copy"]
    leads = fs["QP-LEAK-002"].evidence["features"]
    assert {(d["feature"], d["lead"]) for d in leads} == {("future3", 3)}


def test_clean_features_pass(prices):
    r = prices["close"].pct_change()
    feats = pd.DataFrame({"lag1": r.shift(1), "mom": prices["close"].pct_change(20)})
    fs = feature_leakage_findings(feats, target=(r.shift(-1) > 0).astype(float), asset_returns=r)
    assert all(f.severity is Severity.PASS for f in fs)


def test_signal_foresight(prices):
    r = prices["close"].pct_change()
    perfect = np.sign(r.shift(-1))
    ev, f = signal_foresight_finding(perfect, r)
    assert f.severity is Severity.WARN and ev["hit_rate"] == 1.0
    ev2, f2 = signal_foresight_finding(np.sign(r), r)
    assert f2.severity is Severity.PASS and 0.3 < ev2["hit_rate"] < 0.7
