import numpy as np
import pandas as pd


def generate_signals(data: pd.DataFrame) -> pd.Series:
    close = data["close"].to_numpy()
    out = np.zeros(len(close))
    for i in range(len(close) - 1):
        signal = 1.0 if close[i + 1] > close[i] else 0.0
        out[i] = signal
    return pd.Series(out, index=data.index)
