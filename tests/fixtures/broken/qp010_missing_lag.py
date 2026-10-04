import pandas as pd


def backtest(df: pd.DataFrame) -> pd.Series:
    df["returns"] = df["close"].pct_change()
    df["signal"] = (df["close"] > df["close"].rolling(20).mean()).astype(float)
    df["fee"] = 0.0001 * df["signal"].diff().abs()
    df["strategy"] = df["signal"] * df["returns"] - df["fee"]
    return df["strategy"]
