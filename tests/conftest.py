"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantproof.config import AuditConfig
from quantproof.data.synthetic import generate_prices

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXAMPLES = ROOT / "examples"


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return generate_prices(600, seed=3, ar1=-0.1)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def quick_config() -> AuditConfig:
    cfg = AuditConfig(quick=True)
    cfg.statistics.n_bootstrap = 200
    return cfg
