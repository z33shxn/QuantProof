"""DELIBERATELY BROKEN example: feature and normalization leakage in an ML workflow.

Typical mistakes from a notebook that "worked":
1. The scaler is fitted on the whole dataset before splitting (QP006).
2. train_test_split shuffles time-ordered rows (QP004).
3. The forward-return column used to build the label is left in the feature list (QP012).
4. The model is trained once on (shuffled) data spanning the whole sample, so
   past predictions depend on future observations (runtime causality FAIL).

Requires scikit-learn (pip install "quantproof[ml]").
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NAME = "leaky_logistic"
PARAMETERS = {"seed": 0}
EXECUTION = {"signal_lag": 1, "commission_bps": 1.0, "spread_bps": 2.0, "slippage_bps": 2.0}
FEATURES = ["ret_1", "ret_5", "ret_20", "vol_20", "fwd_ret"]


def generate_signals(data: pd.DataFrame, seed: int = 0) -> pd.Series:
    df = data.copy()
    ret = df["close"].pct_change()
    df["ret_1"] = ret
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)
    df["vol_20"] = ret.rolling(20).std()
    df["fwd_ret"] = ret.shift(-1)
    df["target"] = (df["fwd_ret"] > 0).astype(int)
    df = df.dropna()

    X = df[FEATURES]
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=seed
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    _ = model.score(X_test, y_test)  # the "excellent" hold-out accuracy that hid the leak

    proba = model.predict_proba(X_scaled)[:, 1]
    signal = pd.Series(np.where(proba > 0.5, 1.0, -1.0), index=df.index)
    return signal.reindex(data.index)
