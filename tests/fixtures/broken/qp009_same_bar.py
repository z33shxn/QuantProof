import pandas as pd


def backtest(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change()
    signal = (df["close"] > df["close"].rolling(20).mean()).astype(float)
    commission_bps = 1.0
    pnl = signal.shift(1) * returns - signal.diff().abs() * commission_bps / 1e4
    return pnl
