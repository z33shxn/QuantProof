# Strategy contract

```python
import pandas as pd

NAME = "my_strategy"  # optional identifier
PARAMETERS = {"lookback": 20}  # optional defaults (plain literals)
PARAM_GRID = {"lookback": [10, 20, 40]}  # optional: every variant tried
EXECUTION = {  # optional: your own assumptions
    "signal_lag": 1,  # bars between the signal bar's close and the fill
    "fill": "close",  # "close" or "next_open"
    "commission_bps": 1.0,
    "spread_bps": 2.0,  # full quoted spread; half is paid per trade
    "slippage_bps": 2.0,
    "impact_coefficient": 0.0,
    "capital": 1_000_000,
}


def generate_signals(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Target weight for every bar, decided with data up to and including that bar."""
    ...
```

Rules:

- `data` is the prepared price frame (DatetimeIndex, normalized lower-case columns such as
  `open`, `high`, `low`, `close`, `volume`). The function receives a copy.
- Return a Series aligned to `data.index` (a subset of the index is reindexed, NaN means
  "no position yet"), a one-column DataFrame, or a 1-D array of the same length.
- Values are target weights (fractions of equity, e.g. −1 … 1). The reference simulator
  handles timing and costs; do not shift the output yourself.
- Keep metadata as literals so the static analyzer can read it.
- The function must be deterministic (seed any randomness) for the causality test to be
  conclusive.
- Unknown `EXECUTION` keys are rejected.

Callables are accepted directly: `audit(strategy=my_function, data=prices)`.

**Security:** loading a strategy file executes it in your Python process. Only audit code
you trust or use an isolated environment. See [SECURITY.md](../../SECURITY.md).
