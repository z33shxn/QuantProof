"""Generate the synthetic datasets used by the examples (deterministic, offline).

    python examples/data/generate.py

prices.parquet / prices.csv
    1,500 business days of a two-regime Markov-switching price process with mild
    negative first-order autocorrelation (a stand-in for bid-ask bounce). See
    quantproof.data.synthetic for the model.
noise.parquet
    A driftless random walk: no rule can have genuine skill on it.

All data is synthetic; no third-party market data is redistributed.
"""

from __future__ import annotations

from pathlib import Path

from quantproof.data import generate_prices

HERE = Path(__file__).resolve().parent


def main() -> None:
    prices = generate_prices(1500, seed=2026, ar1=-0.15, start="2019-01-01")
    prices.to_parquet(HERE / "prices.parquet")
    prices.round(6).to_csv(HERE / "prices.csv")
    noise = generate_prices(1500, seed=99, mu=0.0, stress_vol_multiplier=1.0, start="2019-01-01")
    noise.to_parquet(HERE / "noise.parquet")
    print(f"wrote {HERE / 'prices.parquet'}, {HERE / 'prices.csv'}, {HERE / 'noise.parquet'}")


if __name__ == "__main__":
    main()
