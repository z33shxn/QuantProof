"""Audit configuration.

Configuration can be built in Python or loaded from YAML, TOML, or JSON with
:meth:`AuditConfig.from_file`. Unknown keys are rejected so a typo such as
``comission_bps`` fails loudly instead of being silently ignored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from quantproof.errors import QuantProofConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataValidationConfig(_Strict):
    """Thresholds for data-quality checks (``QP-DATA-*``)."""

    enabled: bool = True
    require_ohlc: bool = False
    allow_duplicate_timestamps: bool = False
    max_gap_multiple: float = Field(
        5.0, gt=1.0, description="Flag gaps longer than this multiple of the median spacing."
    )
    max_stale_run: int = Field(
        5, ge=2, description="Flag runs of identical closes at least this long (forward fill)."
    )
    extreme_return_threshold: float = Field(
        0.5, gt=0, description="Flag single-period absolute returns above this fraction."
    )
    disabled_rules: list[str] = Field(default_factory=list)


class ExecutionConfig(_Strict):
    """Execution assumptions applied by the reference simulator.

    These are the *auditor's* realistic assumptions. A strategy may declare its
    own (often optimistic) assumptions via a module-level ``EXECUTION`` dict;
    the audit compares both.
    """

    signal_lag: int = Field(
        1,
        ge=0,
        description=(
            "Bars between the bar whose close generated the signal and the bar at whose "
            "close the trade is filled. 0 = fill at the same close that produced the signal."
        ),
    )
    fill: Literal["close", "next_open"] = "close"
    commission_bps: float = Field(1.0, ge=0)
    spread_bps: float = Field(2.0, ge=0, description="Full quoted spread; half is paid per trade.")
    slippage_bps: float = Field(2.0, ge=0)
    impact_coefficient: float = Field(
        0.0, ge=0, description="Square-root impact coefficient Y; 0 disables impact."
    )
    capital: float = Field(1_000_000.0, gt=0)
    cost_grid_bps: list[float] = Field(
        default_factory=lambda: [0.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        description="Round-trip-agnostic one-way cost levels for the cost-sensitivity table.",
    )

    @property
    def one_way_cost_bps(self) -> float:
        """Commission + half spread + slippage, in basis points of traded notional."""
        return self.commission_bps + self.spread_bps / 2.0 + self.slippage_bps


class StatisticsConfig(_Strict):
    """Settings for statistical diagnostics."""

    periods_per_year: float = Field(252.0, gt=0)
    risk_free_rate: float = Field(0.0, description="Annual risk-free rate (decimal).")
    trials: int | None = Field(
        None,
        ge=1,
        description=(
            "Number of strategy variants tried before this one was selected. Used by the "
            "Deflated Sharpe Ratio. If omitted, the size of the strategy's PARAM_GRID is used, "
            "or 1 if there is no grid (which understates selection bias)."
        ),
    )
    benchmark_sharpe: float = Field(0.0, description="Annualized Sharpe hurdle for the PSR.")
    confidence: float = Field(0.95, gt=0.5, lt=1.0)
    n_bootstrap: int = Field(1000, ge=100)
    bootstrap_method: Literal["stationary", "circular", "iid"] = "stationary"
    block_length: float | None = Field(
        None, gt=0, description="Mean block length; None = Politis-White automatic selection."
    )
    max_plausible_sharpe: float = Field(
        4.0, gt=0, description="Annualized Sharpe above which results are flagged as implausible."
    )
    min_observations: int = Field(60, ge=10)


class ValidationConfig(_Strict):
    """Temporal validation and overfitting diagnostics."""

    walk_forward: bool = True
    train_fraction: float = Field(0.5, gt=0, lt=1, description="Initial train window share.")
    n_test_windows: int = Field(5, ge=2)
    expanding: bool = True
    gap: int = Field(0, ge=0, description="Bars dropped between train and test windows.")
    pbo: bool = True
    pbo_partitions: int = Field(10, ge=4, description="Even number of CSCV blocks (S).")
    pbo_max_combinations: int = Field(5000, ge=10)
    cpcv: bool = True
    cpcv_groups: int = Field(6, ge=3)
    cpcv_test_groups: int = Field(2, ge=1)
    embargo: float = Field(0.01, ge=0, lt=0.5, description="Embargo as a fraction of samples.")
    reality_check: bool = True
    max_trials: int = Field(
        200, ge=2, description="Cap on grid evaluations; larger grids are sub-sampled."
    )

    @model_validator(mode="after")
    def _check(self) -> ValidationConfig:
        if self.pbo_partitions % 2:
            raise ValueError("pbo_partitions must be even (CSCV splits blocks in half).")
        if self.cpcv_test_groups >= self.cpcv_groups:
            raise ValueError("cpcv_test_groups must be smaller than cpcv_groups.")
        return self


SchemeName = Literal["additive", "multiplicative", "permutation", "shock"]


def _all_schemes() -> list[SchemeName]:
    return ["additive", "multiplicative", "permutation", "shock"]


class CausalityConfig(_Strict):
    """Runtime future-perturbation test."""

    enabled: bool = True
    n_timestamps: int = Field(8, ge=1)
    schemes: list[SchemeName] = Field(default_factory=_all_schemes)
    magnitude: float = Field(
        0.5, gt=0, description="Perturbation scale, in units of each column's return std."
    )
    min_history_fraction: float = Field(
        0.2, ge=0, lt=1, description="Earliest test timestamp as a fraction of the sample."
    )
    rtol: float = Field(1e-9, ge=0)
    atol: float = Field(1e-12, ge=0)


class RegimeConfig(_Strict):
    """Regime definitions. All regime labels use only information up to t-1."""

    enabled: bool = True
    vol_window: int = Field(63, ge=5)
    vol_quantiles: list[float] = Field(default_factory=lambda: [1 / 3, 2 / 3])
    vol_labels: list[str] = Field(default_factory=lambda: ["low", "normal", "high"])
    drawdown_thresholds: list[float] = Field(
        default_factory=lambda: [-0.10, -0.20],
        description="Benchmark drawdown cut-offs separating normal / drawdown / deep drawdown.",
    )
    trend_window: int = Field(126, ge=5, description="Trailing window for bull/bear labels.")
    min_observations: int = Field(40, ge=5)

    @model_validator(mode="after")
    def _check(self) -> RegimeConfig:
        if len(self.vol_labels) != len(self.vol_quantiles) + 1:
            raise ValueError("vol_labels must have exactly one more entry than vol_quantiles.")
        if sorted(self.vol_quantiles) != self.vol_quantiles or not all(
            0 < q < 1 for q in self.vol_quantiles
        ):
            raise ValueError("vol_quantiles must be increasing values in (0, 1).")
        if sorted(self.drawdown_thresholds, reverse=True) != self.drawdown_thresholds:
            raise ValueError("drawdown_thresholds must be decreasing (e.g. [-0.1, -0.2]).")
        return self


class SensitivityConfig(_Strict):
    """Parameter-surface robustness thresholds."""

    enabled: bool = True
    robust_ratio: float = Field(0.7, gt=0, le=1)
    fragile_ratio: float = Field(0.4, gt=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> SensitivityConfig:
        if self.fragile_ratio >= self.robust_ratio:
            raise ValueError("fragile_ratio must be smaller than robust_ratio.")
        return self


class StaticConfig(_Strict):
    """Static analyzer options."""

    enabled: bool = True
    disabled_rules: list[str] = Field(default_factory=list)
    max_trials_threshold: int = Field(
        100, ge=2, description="Hyper-parameter search size above which QP013 warns."
    )


class AuditConfig(_Strict):
    """Top-level audit configuration."""

    seed: int = 42
    quick: bool = Field(
        False, description="Reduce bootstrap/CSCV/perturbation sizes for fast iteration."
    )
    data: DataValidationConfig = Field(default_factory=DataValidationConfig)
    static: StaticConfig = Field(default_factory=StaticConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    causality: CausalityConfig = Field(default_factory=CausalityConfig)
    regimes: RegimeConfig = Field(default_factory=RegimeConfig)
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)

    def effective(self) -> AuditConfig:
        """Return a copy with ``quick`` reductions applied (explicit, documented)."""
        if not self.quick:
            return self.model_copy(deep=True)
        cfg = self.model_copy(deep=True)
        cfg.statistics.n_bootstrap = min(cfg.statistics.n_bootstrap, 200)
        cfg.validation.pbo_max_combinations = min(cfg.validation.pbo_max_combinations, 500)
        cfg.validation.max_trials = min(cfg.validation.max_trials, 50)
        cfg.causality.n_timestamps = min(cfg.causality.n_timestamps, 4)
        return cfg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditConfig:
        """Validate a nested mapping into a config, with readable errors."""
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            lines = []
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                lines.append(f"  - {loc}: {err['msg']}")
            raise QuantProofConfigError(
                "Invalid QuantProof configuration:\n" + "\n".join(lines)
            ) from exc

    @classmethod
    def from_file(cls, path: str | Path) -> AuditConfig:
        """Load configuration from a ``.yaml``/``.yml``, ``.toml`` or ``.json`` file."""
        p = Path(path)
        if not p.exists():
            raise QuantProofConfigError(f"Configuration file not found: {p}")
        text = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()
        data: Any
        if suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(text) or {}
        elif suffix == ".toml":
            if sys.version_info >= (3, 11):
                import tomllib
            else:  # pragma: no cover - exercised on Python 3.10
                import tomli as tomllib
            data = tomllib.loads(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise QuantProofConfigError(
                f"Unsupported configuration format {suffix!r}; use .yaml, .toml or .json."
            )
        if not isinstance(data, dict):
            raise QuantProofConfigError(f"Configuration root in {p} must be a mapping.")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary (JSON-compatible)."""
        return self.model_dump(mode="json")
