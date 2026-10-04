import pandas as pd


def generate_signals(data: pd.DataFrame) -> pd.Series:
    smooth = data["close"].rolling(window=20, center=True).mean()
    return (data["close"] > smooth).astype(float)
