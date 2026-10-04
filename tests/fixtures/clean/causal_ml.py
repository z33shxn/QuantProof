"""A correctly structured ML workflow: no rule should fire as WARN/FAIL."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quantproof.validation import PurgedKFold


def research(df: pd.DataFrame) -> list[float]:
    df["ret_1"] = df["close"].pct_change()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["label"] = (df["close"].pct_change().shift(-1) > 0).astype(int)
    data = df.dropna()
    X = data[["ret_1", "vol_20"]]
    y = data["label"]
    scores = []
    for train, test in PurgedKFold(5, label_horizon=1, embargo=0.01).split(X):
        model = make_pipeline(StandardScaler(), LogisticRegression())
        model.fit(X.iloc[train], y.iloc[train])
        scores.append(model.score(X.iloc[test], y.iloc[test]))
    return scores
