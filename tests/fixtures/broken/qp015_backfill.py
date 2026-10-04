import pandas as pd


def generate_signals(data: pd.DataFrame) -> pd.Series:
    close = data["close"].bfill()
    trend = close.interpolate()
    return (close > trend.rolling(10).mean()).astype(float)
