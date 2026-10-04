"""Generate the synthetic datasets used by the examples (deterministic, offline).

    python examples/data/generate.py

prices.parquet / prices.csv
    1,500 business days of a two-regime Markov-switching price process with mild
    negative first-order autocorrelation (a stand-in for bid-ask bounce). See
    quantproof.data.synthetic for the model.
noise.parquet
    A driftless random walk: no rule can have genuine skill on it.
universe.parquet
    10 symbols x 1,250 business days in long format (timestamp, symbol, OHLCV) with a
    common market factor and slowly varying per-symbol drifts: a weak cross-sectional
    momentum effect exists by construction (see quantproof.data.synthetic).

All data is synthetic; no third-party market data is redistributed.
"""

from __future__ import annotations

from pathlib import Path

from quantproof.data import generate_prices, generate_universe

HERE = Path(__file__).resolve().parent


def main() -> None:
    prices = generate_prices(1500, seed=2026, ar1=-0.15, start="2019-01-01")
    prices.to_parquet(HERE / "prices.parquet")
    prices.round(6).to_csv(HERE / "prices.csv")
    noise = generate_prices(1500, seed=99, mu=0.0, stress_vol_multiplier=1.0, start="2019-01-01")
    noise.to_parquet(HERE / "noise.parquet")
    universe = generate_universe(10, 1250, seed=7, start="2019-01-01")
    num = universe.select_dtypes("float").columns
    universe[num] = universe[num].round(4)
    universe.to_parquet(HERE / "universe.parquet", index=False)
    print(f"wrote prices, noise and universe datasets to {HERE}")


if __name__ == "__main__":
    main()
