import pandas as pd


def generate_signals(data: pd.DataFrame) -> pd.Series:
    future = data["close"].shift(-1)
    signal = (future > data["close"]).astype(float)
    return signal
