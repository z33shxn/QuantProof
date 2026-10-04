"""Data loading, validation, and synthetic example data."""

from quantproof.data.loaders import load_frame, load_prices, load_returns, prepare_prices
from quantproof.data.synthetic import generate_prices, generate_universe
from quantproof.data.validation import DATA_RULES, validate_data

__all__ = [
    "DATA_RULES",
    "generate_prices",
    "generate_universe",
    "load_frame",
    "load_prices",
    "load_returns",
    "prepare_prices",
    "validate_data",
]
