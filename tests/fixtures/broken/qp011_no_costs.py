import pandas as pd


def backtest(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change()
    position = (df["close"] > df["close"].rolling(50).mean()).astype(float)
    # Assumption: market-on-close execution at the next bar.
    return position.shift(1) * returns
