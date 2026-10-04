import pandas as pd


def generate_signals(data: pd.DataFrame) -> pd.Series:
    z = (data["close"] - data["close"].mean()) / data["close"].std()
    return (z < 0).astype(float)
