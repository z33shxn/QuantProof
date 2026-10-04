"""Audit orchestration.

:func:`audit` runs every applicable analyzer and assembles an :class:`AuditResult`.
Two input modes are supported:

1. **Strategy mode** — a strategy (file path or callable following the contract in
   :mod:`quantproof.strategy`) plus price data. All checks run: data validation,
   static analysis, future-perturbation causality test, leakage diagnostics,
   execution realism, statistics, selection-bias diagnostics over ``PARAM_GRID``,
   regimes, parameter sensitivity, and the reproducibility manifest.
2. **Results mode** — pre-computed outputs from any backtester via
   :class:`~quantproof.adapters.ResearchArtifacts` (or the ``returns=`` shortcut).
   Code-level checks are reported as not run.
"""

from __future__ import annotations

import inspect
import math
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantproof.adapters.base import ResearchArtifacts
from quantproof.analyzers.causal.runner import causality_findings, run_causality_test
from quantproof.analyzers.execution.analyzer import analyze_execution, metrics, realistic_simulation
from quantproof.analyzers.leakage.detector import feature_leakage_findings, signal_foresight_finding
from quantproof.analyzers.static.analyzer import analyze_file, analyze_source
from quantproof.analyzers.statistical.analyzer import analyze_statistics
from quantproof.analyzers.statistical.selection import (
    analyze_selection,
    oos_evidence,
    oos_finding,
    walk_forward_single,
)
from quantproof.config import AuditConfig
from quantproof.data.loaders import DataSource, describe_source, load_frame, prepare_prices
from quantproof.data.panel import is_panel, symbols, unique_times, wide
from quantproof.data.schemas import primary_price_column
from quantproof.data.validation import validate_data
from quantproof.errors import QuantProofDataError, QuantProofInputError, QuantProofStrategyError
from quantproof.execution.timing import analyze_execution_timing
from quantproof.execution.turnover import turnover_stats
from quantproof.experiments.hashing import hash_dataframe, hash_file, schema_of
from quantproof.experiments.lineage import Lineage
from quantproof.experiments.manifest import build_manifest
from quantproof.regimes.analysis import regime_analysis
from quantproof.results import AuditResult, Category, Finding
from quantproof.sensitivity.parameter_analysis import (
    analyze_parameter_surface,
    render_ascii_surface,
    sensitivity_findings,
    surface_matrix,
)
from quantproof.severity import Confidence, Severity
from quantproof.statistics.sharpe import sharpe_ratios
from quantproof.strategy import StrategySpec, load_strategy

LIMITATIONS = [
    "A PASS verdict means the configured checks found no problem; it does not show that a "
    "strategy is profitable or will work in live trading.",
    "Static analysis is heuristic: it reasons about code patterns and can miss leakage hidden "
    "behind indirection, or flag valid code in unusual contexts.",
    "The future-perturbation test provides evidence against look-ahead for the tested "
    "timestamps and schemes; it cannot detect leakage already embedded in the input data "
    "(survivorship bias, restated fundamentals, mislabelled timestamps).",
    "Execution simulation uses bar prices with simple cost models; it does not reproduce "
    "an exchange matching engine, queue position, or intrabar latency.",
    "PSR/DSR rely on asymptotic normality of the Sharpe estimator with i.i.d. returns; "
    "bootstrap intervals are approximate.",
    "Strategy code is executed in the current Python process without sandboxing.",
]


def _downsample(s: pd.Series, max_points: int = 400) -> dict[str, list[Any]]:
    s = s.dropna()
    if len(s) > max_points:
        idx = np.unique(np.linspace(0, len(s) - 1, max_points).round().astype(int))
        s = s.iloc[idx]
    return {
        "x": [
            pd.Timestamp(i).isoformat() if isinstance(i, pd.Timestamp) else str(i) for i in s.index
        ],
        "y": [float(v) for v in s.to_numpy()],
    }


def _equity(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()


def _not_run(rule_id: str, category: str, title: str, reason: str) -> Finding:
    return Finding(
        id=rule_id,
        category=category,
        severity=Severity.INFO,
        title=f"{title} not run",
        message=reason,
    )


def _static_findings(spec: StrategySpec, cfg: AuditConfig) -> tuple[list[Finding], dict[str, Any]]:
    if not cfg.static.enabled:
        return [], {"enabled": False}
    if (
        spec.source_path is not None
        and spec.source_path.suffix == ".py"
        and spec.source_path.exists()
    ):
        source_path = spec.source_path
        try:
            func_file = Path(inspect.getfile(spec.func)).resolve()
        except (TypeError, OSError):
            func_file = source_path.resolve()
        if func_file == source_path.resolve() and spec.func.__name__ != "generate_signals":
            # A callable defined in a larger file: analyze only its source.
            src = textwrap.dedent(inspect.getsource(spec.func))
            found = analyze_source(
                src,
                f"{source_path}:{spec.func.__name__}",
                cfg.static,
                entry_points={spec.func.__name__},
            )
            return found, {"enabled": True, "target": str(source_path), "scope": spec.func.__name__}
        return analyze_file(source_path, cfg.static), {"enabled": True, "target": str(source_path)}
    try:
        src = textwrap.dedent(inspect.getsource(spec.func))
    except (TypeError, OSError):
        return [
            _not_run(
                "QP-STATIC",
                Category.STATIC,
                "Static analysis",
                "Strategy source code is not available.",
            )
        ], {"enabled": False}
    found = analyze_source(
        src, f"<{spec.name}>", cfg.static, entry_points={getattr(spec.func, "__name__", "")}
    )
    return found, {"enabled": True, "target": spec.name}


def _first_valid(s: pd.Series | pd.DataFrame) -> Any:
    idx = s.first_valid_index()
    return idx


def _grid_matrix(
    spec: StrategySpec,
    prices: pd.DataFrame,
    cfg: AuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Any]:
    """Realistic net returns of every grid configuration, aligned on a common window."""
    combos = spec.grid(cfg.validation.max_trials, seed=cfg.seed)
    columns: dict[str, pd.Series] = {}
    params_rows = []
    errors: list[str] = []
    starts = []
    for combo in combos:
        label = ",".join(f"{k}={v}" for k, v in combo.items())
        try:
            sig = spec(prices, **combo)
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        start = _first_valid(sig)
        if start is None:
            errors.append(f"{label}: all signals are NaN")
            continue
        starts.append(start)
        sim = realistic_simulation(prices, sig, cfg.execution, cfg.statistics.periods_per_year)
        columns[label] = sim.net_returns
        params_rows.append({"label": label, **combo})
    if not columns:
        return pd.DataFrame(), pd.DataFrame(), errors, None
    start = max(starts)
    matrix = pd.DataFrame(columns).loc[start:]
    # Returns at the first bar after warm-up are 0 because positions start flat.
    return matrix, pd.DataFrame(params_rows), errors, start


def _sensitivity(
    matrix: pd.DataFrame, params: pd.DataFrame, cfg: AuditConfig
) -> tuple[dict[str, Any], list[Finding]]:
    ppy = cfg.statistics.periods_per_year
    names = [c for c in params.columns if c != "label" and params[c].nunique() > 1]
    if not names or not all(pd.api.types.is_numeric_dtype(params[c]) for c in names):
        return {"note": "Parameter surface requires numeric parameters with ≥2 values."}, []
    m = matrix.to_numpy(dtype=float)
    half = m.shape[0] // 2
    res = params.copy()
    res["sharpe"] = sharpe_ratios(m) * math.sqrt(ppy)
    res["is_sharpe"] = sharpe_ratios(m[:half]) * math.sqrt(ppy)
    res["oos_sharpe"] = sharpe_ratios(m[half:]) * math.sqrt(ppy)
    full = analyze_parameter_surface(
        res,
        names,
        "sharpe",
        robust_ratio=cfg.sensitivity.robust_ratio,
        fragile_ratio=cfg.sensitivity.fragile_ratio,
    )
    isoos = analyze_parameter_surface(
        res,
        names,
        "is_sharpe",
        oos_metric="oos_sharpe",
        robust_ratio=cfg.sensitivity.robust_ratio,
        fragile_ratio=cfg.sensitivity.fragile_ratio,
    )
    findings = sensitivity_findings(full)[:1]
    findings += [f for f in sensitivity_findings(isoos) if f.id == "QP-SENS-002"]
    section: dict[str, Any] = {
        "parameters": names,
        "surface": full.to_dict(),
        "is_oos": {
            "split": "first half (in-sample) vs second half (out-of-sample)",
            "spearman": isoos.is_oos_spearman,
            "oos_at_is_best": isoos.oos_at_is_best,
            "oos_median": isoos.oos_median,
            "is_best": isoos.best,
        },
        "table": res.drop(columns=["label"]).to_dict(orient="records"),
    }
    if len(names) >= 2:
        x, y = names[0], names[1]
        mat = surface_matrix(res.drop(columns=["label", "is_sharpe", "oos_sharpe"]), x, y, "sharpe")
        section["heatmap"] = {
            "x": x,
            "y": y,
            "x_values": [float(v) for v in mat.columns],
            "y_values": [float(v) for v in mat.index],
            "values": [
                [None if not np.isfinite(v) else float(v) for v in row] for row in mat.to_numpy()
            ],
        }
        section["ascii"] = render_ascii_surface(
            res.drop(columns=["label", "is_sharpe", "oos_sharpe"]), x, y, "sharpe"
        )
    return section, findings


def _manifest(
    *,
    spec: StrategySpec | None,
    data_source: Any,
    raw_hash: str | None,
    prices: pd.DataFrame | None,
    cfg: AuditConfig,
    sections: dict[str, Any],
    lineage: Lineage,
    extra_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strategy_info: dict[str, Any] = {"name": spec.name if spec else None}
    git_path = None
    if spec is not None and spec.source_path is not None and spec.source_path.exists():
        strategy_info["path"] = str(spec.source_path)
        strategy_info["sha256"] = hash_file(spec.source_path)
        git_path = spec.source_path
    data_info: dict[str, Any] = {
        "source": describe_source(data_source) if data_source is not None else None
    }
    if isinstance(data_source, (str, Path)) and Path(data_source).exists():
        data_info["file_sha256"] = hash_file(data_source)
        data_info["format"] = Path(data_source).suffix.lstrip(".")
    if raw_hash:
        data_info["content_sha256"] = raw_hash
    if prices is not None:
        data_info["schema"] = schema_of(prices)
    data_info.update(extra_data or {})
    validation = {
        "walk_forward": {
            "enabled": cfg.validation.walk_forward,
            "train_fraction": cfg.validation.train_fraction,
            "n_test_windows": cfg.validation.n_test_windows,
            "expanding": cfg.validation.expanding,
            "gap": cfg.validation.gap,
        },
        "pbo": {"enabled": cfg.validation.pbo, "partitions": cfg.validation.pbo_partitions},
        "cpcv": {
            "enabled": cfg.validation.cpcv,
            "groups": cfg.validation.cpcv_groups,
            "test_groups": cfg.validation.cpcv_test_groups,
            "embargo": cfg.validation.embargo,
        },
    }
    execution = {
        "signal_lag": cfg.execution.signal_lag,
        "fill": cfg.execution.fill,
        "commission_bps": cfg.execution.commission_bps,
        "spread_bps": cfg.execution.spread_bps,
        "slippage_bps": cfg.execution.slippage_bps,
        "impact_coefficient": cfg.execution.impact_coefficient,
        "declared_by_strategy": spec.execution if spec else None,
    }
    return build_manifest(
        strategy=strategy_info,
        data=data_info,
        config=cfg.to_dict(),
        validation=validation,
        execution=execution,
        seed=cfg.seed,
        parameters={
            "defaults": spec.parameters if spec else {},
            "grid": spec.param_grid if spec else {},
            "grid_size": spec.grid_size if spec else 0,
        },
        lineage=lineage.to_list(),
        git_path=git_path,
    )


def _load_prices(
    data: DataSource,
    cfg: AuditConfig,
    timestamp_column: str | None,
    lineage: Lineage,
    symbol_column: str | None = None,
) -> tuple[pd.DataFrame, list[Finding], str, dict[str, Any]]:
    raw = load_frame(data, timestamp_column=timestamp_column)
    raw_hash = hash_dataframe(raw)
    findings = validate_data(
        raw, cfg.data, timestamp_column=timestamp_column, symbol_column=symbol_column
    )
    prepared = prepare_prices(raw, timestamp_column=timestamp_column, symbol_column=symbol_column)
    prices = prepared.prices
    if prepared.actions:
        findings.append(
            Finding(
                id="QP-DATA-000",
                category=Category.DATA,
                severity=Severity.INFO,
                title="Data preparation actions",
                message=" ".join(prepared.actions),
                evidence={"actions": prepared.actions},
            )
        )
    if primary_price_column(prices.columns) is None:
        raise QuantProofDataError(
            "No price column found ('close', 'adj_close' or 'price'). Rename your price column."
        )
    lineage.record(
        "load_and_prepare_data",
        inputs={"raw": raw_hash},
        outputs={"prices": hash_dataframe(prices)},
        notes=prepared.actions,
    )
    times = unique_times(prices)
    section: dict[str, Any] = {
        "rows_raw": len(raw),
        "rows_used": len(prices),
        "start": times[0].isoformat(),
        "end": times[-1].isoformat(),
        "columns": [str(c) for c in prices.columns],
        "actions": prepared.actions,
    }
    if is_panel(prices):
        section["symbols"] = symbols(prices)
        section["n_timestamps"] = len(times)
    return prices, findings, raw_hash, section


def _asset_returns(prices: pd.DataFrame, price_col: str) -> tuple[pd.DataFrame | pd.Series, str]:
    """Per-asset returns (wide for panels) and a description of the reference series."""
    if is_panel(prices):
        return (
            wide(prices, price_col).pct_change(fill_method=None),
            "equal-weight average of the universe's returns",
        )
    return prices[price_col].pct_change(fill_method=None), "the traded asset's returns"


def _reference_returns(asset_returns: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(asset_returns, pd.DataFrame):
        return asset_returns.mean(axis=1, skipna=True)
    return asset_returns


def audit(
    strategy: str | Path | Callable[..., Any] | StrategySpec | None = None,
    data: DataSource | None = None,
    *,
    config: AuditConfig | None = None,
    benchmark: pd.Series | DataSource | None = None,
    artifacts: ResearchArtifacts | None = None,
    returns: pd.Series | None = None,
    trial_returns: pd.DataFrame | None = None,
    timestamp_column: str | None = None,
    symbol_column: str | None = None,
) -> AuditResult:
    """Audit a strategy (with data) or pre-computed research artifacts.

    Parameters
    ----------
    strategy:
        Path to a strategy file, a callable ``f(data, **params)``, or a :class:`StrategySpec`.
    data:
        Price data: CSV/Parquet path or DataFrame (needs a timestamp and a ``close`` column).
    config:
        :class:`AuditConfig`; defaults are documented in :mod:`quantproof.config`.
    benchmark:
        Optional benchmark returns (Series or file) used for regime analysis.
        Multi-asset data is long format with a symbol column (``symbol``, ``ticker`` …) or a
        ``(timestamp, symbol)`` MultiIndex; see :mod:`quantproof.data.panel`.
    artifacts / returns / trial_returns:
        Results-mode inputs from any backtesting engine.
    timestamp_column / symbol_column:
        Column names when they cannot be detected automatically.
    """
    cfg = (config or AuditConfig()).effective()
    if artifacts is None and (returns is not None or trial_returns is not None):
        artifacts = ResearchArtifacts(returns=returns, trial_returns=trial_returns)
    if strategy is None:
        if artifacts is None:
            raise QuantProofInputError(
                "Nothing to audit: pass strategy=... and data=..., or artifacts/returns=..."
            )
        return _audit_artifacts(artifacts, cfg, benchmark=benchmark, data=data)
    return _audit_strategy(
        strategy,
        data,
        cfg,
        benchmark=benchmark,
        timestamp_column=timestamp_column,
        symbol_column=symbol_column,
    )


def _benchmark_series(benchmark: Any, index: pd.Index) -> pd.Series | None:
    if benchmark is None:
        return None
    if isinstance(benchmark, pd.Series):
        b = benchmark
    else:
        from quantproof.data.loaders import load_returns

        b = load_returns(benchmark)
    return b.astype(float).reindex(index)


def _audit_strategy(
    strategy: Any,
    data: DataSource | None,
    cfg: AuditConfig,
    *,
    benchmark: Any,
    timestamp_column: str | None,
    symbol_column: str | None = None,
) -> AuditResult:
    lineage = Lineage()
    findings: list[Finding] = []
    sections: dict[str, Any] = {}
    spec = load_strategy(strategy)
    static, static_meta = _static_findings(spec, cfg)
    findings += static
    sections["static"] = {
        **static_meta,
        "rules_run": sorted({f.id for f in static}),
        "issues": sum(f.is_issue for f in static),
    }
    if data is None:
        findings.append(
            _not_run(
                "QP-RUNTIME",
                Category.CAUSALITY,
                "Runtime checks",
                "No data supplied; only static analysis ran.",
            )
        )
        manifest = _manifest(
            spec=spec,
            data_source=None,
            raw_hash=None,
            prices=None,
            cfg=cfg,
            sections=sections,
            lineage=lineage,
        )
        sections["overview"] = {"strategy": spec.name, "mode": "static-only"}
        return AuditResult.from_findings(
            findings, sections=sections, manifest=manifest, limitations=LIMITATIONS
        )

    prices, data_findings, raw_hash, data_section = _load_prices(
        data, cfg, timestamp_column, lineage, symbol_column
    )
    findings += data_findings
    sections["data"] = data_section
    ppy = cfg.statistics.periods_per_year
    price_col = primary_price_column(prices.columns)
    assert price_col is not None
    asset_returns, reference_label = _asset_returns(prices, price_col)
    reference = _reference_returns(asset_returns)

    try:
        signals = spec(prices)
    except QuantProofStrategyError:
        raise
    except Exception as exc:
        raise QuantProofStrategyError(
            f"{spec.name}: generate_signals failed on the supplied data: {type(exc).__name__}: {exc}"
        ) from exc
    if np.isinf(signals.to_numpy(dtype=float)).any():
        raise QuantProofStrategyError(f"{spec.name}: signals contain infinite values.")
    start = _first_valid(signals)
    if start is None:
        raise QuantProofStrategyError(f"{spec.name}: every signal is NaN.")
    lineage.record(
        "generate_signals",
        inputs={"prices": hash_dataframe(prices)},
        outputs={"signals": hash_dataframe(signals)},
        parameters=spec.parameters,
    )

    # Causality
    if cfg.causality.enabled:
        report = run_causality_test(lambda d: spec(d), prices, cfg.causality, seed=cfg.seed)
        sections["causality"] = report.to_dict()
        findings += causality_findings(
            report, source=str(spec.source_path) if spec.source_path else spec.name
        )

    # Grid (selection bias)
    matrix = pd.DataFrame()
    params_df = pd.DataFrame()
    grid_errors: list[str] = []
    if spec.param_grid:
        matrix, params_df, grid_errors, grid_start = _grid_matrix(spec, prices, cfg)
        if grid_start is not None and grid_start > start:
            start = grid_start
        lineage.record(
            "evaluate_parameter_grid",
            outputs={"trial_returns": hash_dataframe(matrix)} if not matrix.empty else {},
            parameters={
                "grid_size": spec.grid_size,
                "evaluated": matrix.shape[1],
                "errors": len(grid_errors),
            },
        )

    # Leakage (signal foresight)
    foresight, f_finding = signal_foresight_finding(signals.loc[start:], asset_returns.loc[start:])
    sections["leakage"] = {"signal_foresight": foresight}
    findings.append(f_finding)

    # Execution
    exec_section, exec_findings, realistic = analyze_execution(
        prices,
        signals,
        cfg.execution,
        declared=spec.execution,
        periods_per_year=ppy,
        evaluation_start=start,
    )
    sections["execution"] = exec_section
    findings += exec_findings
    net = realistic.net_returns.loc[start:]
    lineage.record(
        "simulate_realistic_execution",
        inputs={"signals": hash_dataframe(signals)},
        outputs={"net_returns": hash_dataframe(net)},
        parameters=cfg.execution.model_dump(),
    )

    # Statistics
    if matrix.shape[1] >= 2:
        trial_sr = sharpe_ratios(matrix.loc[start:].to_numpy(dtype=float))
    else:
        trial_sr = None
    if cfg.statistics.trials is not None:
        n_trials, declared_trials = cfg.statistics.trials, True
        trials_source = "statistics.trials (declared in config)"
    elif spec.param_grid:
        n_trials, declared_trials = spec.grid_size, True
        trials_source = "PARAM_GRID size"
    else:
        n_trials, declared_trials = 1, False
        trials_source = "not declared (assumed 1)"
    stat_section, stat_findings = analyze_statistics(
        net,
        cfg.statistics,
        n_trials=n_trials,
        trials_declared=declared_trials,
        trial_sharpes=trial_sr,
        seed=cfg.seed,
        scenario_sharpes={
            k: v["metrics"]["sharpe"]
            for k, v in exec_section["scenarios"].items()
            if k in ("naive", "declared")
        },
        trials_source=trials_source,
        trial_returns=matrix.loc[start:] if matrix.shape[1] >= 2 else None,
    )
    sections["statistics"] = stat_section
    findings += stat_findings

    # Validation / selection
    if matrix.shape[1] >= 2:
        val_section, val_findings = analyze_selection(
            matrix.loc[start:], cfg.validation, cfg.statistics, seed=cfg.seed
        )
        val_section["grid_errors"] = grid_errors
        sections["validation"] = val_section
        findings += val_findings
        sens_section, sens_findings = _sensitivity(matrix.loc[start:], params_df, cfg)
        sections["sensitivity"] = sens_section
        findings += sens_findings
    else:
        try:
            wf = walk_forward_single(net, cfg.validation, ppy)
            ev = oos_evidence(
                wf["folds"],
                wf["oos_returns"],
                is_sharpe_mean=None,
                periods_per_year=ppy,
                min_observations=cfg.statistics.min_observations,
            )
            sections["validation"] = {
                "walk_forward": {k: v for k, v in wf.items() if k != "oos_returns"},
                "oos_evidence": ev,
                "note": "No PARAM_GRID: selection diagnostics not applicable.",
            }
            findings.append(oos_finding(ev, selection=False))
        except QuantProofInputError as exc:
            sections["validation"] = {"error": str(exc)}

    # Regimes
    bench = _benchmark_series(benchmark, unique_times(prices))
    if cfg.regimes.enabled:
        ref = bench if bench is not None else reference
        reg_section, reg_findings = regime_analysis(
            net,
            ref.loc[start:] if ref is not None else net,
            cfg.regimes,
            periods_per_year=ppy,
            benchmark_supplied=bench is not None,
        )
        if bench is None:
            reg_section.setdefault("definitions", {})["reference"] = (
                f"Regimes are defined on {reference_label}."
            )
        sections["regimes"] = reg_section
        findings += reg_findings

    naive_net = exec_section["scenarios"]["naive"]["metrics"]
    sections["overview"] = {
        "mode": "strategy",
        "strategy": spec.name,
        "strategy_path": str(spec.source_path) if spec.source_path else None,
        "parameters": spec.parameters,
        "grid_size": spec.grid_size,
        "evaluation_start": pd.Timestamp(start).isoformat(),
        "evaluation_end": unique_times(prices)[-1].isoformat(),
        "n_symbols": len(symbols(prices)) if is_panel(prices) else 1,
        "n_observations": len(net),
        "headline": {
            "label": "Naive (same-bar, no costs)",
            **naive_net,
        },
        "realistic": {
            "label": "Audit (lag + costs)",
            **exec_section["scenarios"]["realistic"]["metrics"],
        },
    }
    from quantproof.execution.fills import simulate

    naive_sim = simulate(prices, signals, fill="close", lag=0, periods_per_year=ppy)
    charts = {
        "naive_equity": _downsample(_equity(naive_sim.net_returns.loc[start:])),
        "realistic_equity": _downsample(_equity(net)),
    }
    if bench is not None:
        charts["benchmark_equity"] = _downsample(_equity(bench.loc[start:]))
    else:
        charts["asset_equity"] = _downsample(_equity(reference.loc[start:]))
    sections["charts"] = charts
    manifest = _manifest(
        spec=spec,
        data_source=data,
        raw_hash=raw_hash,
        prices=prices,
        cfg=cfg,
        sections=sections,
        lineage=lineage,
    )
    return AuditResult.from_findings(
        findings, sections=sections, manifest=manifest, limitations=LIMITATIONS
    )


def _audit_artifacts(
    art: ResearchArtifacts, cfg: AuditConfig, *, benchmark: Any, data: DataSource | None
) -> AuditResult:
    art.validate()
    lineage = Lineage()
    findings: list[Finding] = []
    sections: dict[str, Any] = {}
    ppy = cfg.statistics.periods_per_year
    prices = None
    raw_hash = None
    price_source = data if data is not None else art.prices
    if price_source is not None:
        prices, data_findings, raw_hash, data_section = _load_prices(
            price_source, cfg, None, lineage
        )
        findings += data_findings
        sections["data"] = data_section
    findings.append(
        _not_run(
            "QP-STATIC", Category.STATIC, "Static analysis", "No strategy source code supplied."
        )
    )
    findings.append(
        _not_run(
            "QP-CAUSAL-001",
            Category.CAUSALITY,
            "Future-perturbation test",
            "Requires a strategy callable; results-only audits cannot perturb inputs.",
        )
    )
    returns = art.returns
    if returns is None and art.trial_returns is not None:
        best = int(np.nanargmax(sharpe_ratios(art.trial_returns.dropna().to_numpy(dtype=float))))
        returns = art.trial_returns.iloc[:, best]
        sections["selection_note"] = (
            f"returns not given; using best trial column {art.trial_returns.columns[best]!r}"
        )
    if returns is None:
        raise QuantProofInputError("Results-mode audits need returns or trial_returns.")
    returns = returns.astype(float).dropna()
    lineage.record("input_returns", outputs={"returns": hash_dataframe(returns)})

    asset_returns = None
    if prices is not None:
        pc = primary_price_column(prices.columns)
        asset_returns = _reference_returns(_asset_returns(prices, pc)[0]) if pc else None

    if art.trades is not None and {"signal_time", "execution_time"} <= set(art.trades.columns):
        timing = analyze_execution_timing(art.trades["signal_time"], art.trades["execution_time"])
        sections.setdefault("execution", {})["timing"] = timing.to_dict()
        if timing.n_before_signal:
            sev, title = Severity.FAIL, "Trades executed before their signal"
        elif timing.share_instantaneous > 0.5:
            sev, title = Severity.WARN, "Instantaneous execution assumed"
        else:
            sev, title = Severity.PASS, "Execution occurs after signals"
        findings.append(
            Finding(
                id="QP-EXEC-005",
                category=Category.EXECUTION,
                severity=sev,
                title=title,
                message=(
                    f"{timing.n_trades} trades: {timing.n_before_signal} executed before the signal, "
                    f"{timing.n_instantaneous} with zero latency; median latency {timing.median_latency}."
                ),
                evidence=timing.to_dict(),
                why_it_matters=(
                    "Execution before a signal exists is look-ahead; zero latency is an optimistic "
                    "assumption."
                ),
                recommendation="Record realistic execution timestamps and re-run the backtest.",
                confidence=Confidence.HIGH,
            )
        )
    if art.positions is not None:
        pos = art.positions
        trades = pos.diff().fillna(pos)
        sections.setdefault("execution", {})["turnover"] = turnover_stats(
            trades, periods_per_year=ppy
        )
    if art.features is not None:
        findings += feature_leakage_findings(
            art.features, target=art.target, asset_returns=asset_returns
        )
    if art.signals is not None and asset_returns is not None:
        foresight, f_finding = signal_foresight_finding(art.signals, asset_returns)
        sections["leakage"] = {"signal_foresight": foresight}
        findings.append(f_finding)

    trial = art.trial_returns.dropna() if art.trial_returns is not None else None
    trial_sr = (
        sharpe_ratios(trial.to_numpy(dtype=float))
        if trial is not None and trial.shape[1] >= 2
        else None
    )
    if cfg.statistics.trials is not None:
        n_trials, declared = cfg.statistics.trials, True
        trials_source = "statistics.trials (declared in config)"
    elif trial is not None:
        n_trials, declared = trial.shape[1], True
        trials_source = "columns of trial_returns"
    else:
        n_trials, declared = 1, False
        trials_source = "not declared (assumed 1)"
    stat_section, stat_findings = analyze_statistics(
        returns,
        cfg.statistics,
        n_trials=n_trials,
        trials_declared=declared,
        trial_sharpes=trial_sr,
        seed=cfg.seed,
        trials_source=trials_source,
        trial_returns=trial if trial is not None and trial.shape[1] >= 2 else None,
    )
    sections["statistics"] = stat_section
    findings += stat_findings
    if trial is not None and trial.shape[1] >= 2:
        val_section, val_findings = analyze_selection(
            trial, cfg.validation, cfg.statistics, seed=cfg.seed
        )
        sections["validation"] = val_section
        findings += val_findings
        if art.trial_params is not None:
            params = art.trial_params.reset_index(drop=True).copy()
            params.insert(0, "label", [str(c) for c in trial.columns])
            sens_section, sens_findings = _sensitivity(trial, params, cfg)
            sections["sensitivity"] = sens_section
            findings += sens_findings
    if cfg.regimes.enabled:
        bench = _benchmark_series(
            benchmark if benchmark is not None else art.benchmark, returns.index
        )
        ref = (
            bench
            if bench is not None
            else (asset_returns.reindex(returns.index) if asset_returns is not None else returns)
        )
        reg_section, reg_findings = regime_analysis(
            returns, ref, cfg.regimes, periods_per_year=ppy, benchmark_supplied=bench is not None
        )
        if bench is None and asset_returns is None:
            reg_section["definitions"]["note"] = (
                "No benchmark or prices: regimes use the strategy's own returns."
            )
        sections["regimes"] = reg_section
        findings += reg_findings
    sections["overview"] = {
        "mode": "results",
        "strategy": art.metadata.get("name", "results"),
        "n_observations": len(returns),
        "evaluation_start": pd.Timestamp(returns.index[0]).isoformat()
        if isinstance(returns.index, pd.DatetimeIndex) and len(returns)
        else None,
        "evaluation_end": pd.Timestamp(returns.index[-1]).isoformat()
        if isinstance(returns.index, pd.DatetimeIndex) and len(returns)
        else None,
        "headline": {"label": "Supplied returns", **metrics(returns, ppy)},
        "metadata": art.metadata,
    }
    sections["charts"] = {"realistic_equity": _downsample(_equity(returns))}
    manifest = _manifest(
        spec=None,
        data_source=price_source,
        raw_hash=raw_hash,
        prices=prices,
        cfg=cfg,
        sections=sections,
        lineage=lineage,
        extra_data={"returns_sha256": hash_dataframe(returns)},
    )
    return AuditResult.from_findings(
        findings, sections=sections, manifest=manifest, limitations=LIMITATIONS
    )


__all__ = ["LIMITATIONS", "audit"]
