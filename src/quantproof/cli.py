"""Command-line interface.

Exit codes: 0 = completed and the verdict is below the ``--fail-on`` level,
1 = verdict at or above ``--fail-on`` (default FAIL), 2 = usage or input error.
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer

from quantproof._version import __version__
from quantproof.errors import QuantProofError

app = typer.Typer(
    name="quantproof",
    help="QuantProof — trust your backtest before you trust your strategy.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class ReportFormat(str, Enum):
    html = "html"
    json = "json"
    markdown = "markdown"
    text = "text"


class FailOn(str, Enum):
    fail = "fail"
    warn = "warn"
    never = "never"


def _color() -> bool:
    return sys.stdout.isatty()


def _error(msg: str) -> None:
    typer.echo(f"Error: {msg}", err=True)
    raise typer.Exit(2)


def _exit_code(status: str, fail_on: FailOn) -> int:
    if fail_on is FailOn.never:
        return 0
    if fail_on is FailOn.warn:
        return 1 if status in ("WARN", "FAIL") else 0
    return 1 if status == "FAIL" else 0


def _load_config(path: Path | None, quick: bool, seed: int | None) -> Any:
    from quantproof.config import AuditConfig

    cfg = AuditConfig.from_file(path) if path else AuditConfig()
    if quick:
        cfg.quick = True
    if seed is not None:
        cfg.seed = seed
    return cfg


@app.command()
def version() -> None:
    """Print the QuantProof version."""
    typer.echo(f"quantproof {__version__}")


@app.command("audit")
def audit_cmd(
    strategy: Annotated[
        Path | None, typer.Option("--strategy", "-s", help="Strategy .py file.")
    ] = None,
    data: Annotated[
        Path | None, typer.Option("--data", "-d", help="Price data (.csv/.parquet).")
    ] = None,
    returns: Annotated[
        Path | None,
        typer.Option("--returns", help="Results mode: strategy returns (.csv/.parquet)."),
    ] = None,
    trials: Annotated[
        Path | None,
        typer.Option("--trials", help="Results mode: returns of every variant tried (T x N)."),
    ] = None,
    benchmark: Annotated[
        Path | None, typer.Option("--benchmark", "-b", help="Benchmark returns file.")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML/TOML/JSON config.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the report here.")
    ] = None,
    fmt: Annotated[
        ReportFormat | None,
        typer.Option("--format", "-f", help="Report format (default: from extension)."),
    ] = None,
    quick: Annotated[
        bool, typer.Option("--quick", help="Smaller bootstrap/CSCV/perturbation sizes.")
    ] = False,
    seed: Annotated[int | None, typer.Option("--seed", help="Override the random seed.")] = None,
    fail_on: Annotated[
        FailOn, typer.Option("--fail-on", help="Exit 1 at this verdict level.")
    ] = FailOn.fail,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Do not print the summary.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show every finding.")] = False,
) -> None:
    """Audit a strategy (with --strategy and --data) or pre-computed returns (--returns)."""
    from quantproof.audit.engine import audit
    from quantproof.data.loaders import load_frame, load_returns, prepare_prices
    from quantproof.reports import render, write_report
    from quantproof.reports.text import render_text

    try:
        cfg = _load_config(config, quick, seed)
        if strategy is not None:
            result = audit(strategy=strategy, data=data, config=cfg, benchmark=benchmark)
        elif returns is not None or trials is not None:
            trial_df = prepare_prices(load_frame(trials)).prices if trials else None
            result = audit(
                returns=load_returns(returns) if returns else None,
                trial_returns=trial_df,
                data=data,
                config=cfg,
                benchmark=benchmark,
            )
        else:
            _error("Provide --strategy (with --data) or --returns/--trials.")
            return
    except QuantProofError as exc:
        _error(str(exc))
        return
    if output is not None:
        path = write_report(result, output, fmt.value if fmt else None)
        if not quiet:
            typer.echo(f"Report written to {path}", err=True)
    elif fmt is not None and fmt is not ReportFormat.text:
        typer.echo(render(result, fmt.value))
        raise typer.Exit(_exit_code(result.status.value, fail_on))
    if not quiet:
        typer.echo(render_text(result, verbose=verbose, color=_color()))
    raise typer.Exit(_exit_code(result.status.value, fail_on))


@app.command()
def report(
    result_json: Annotated[Path, typer.Argument(help="JSON result written by `quantproof audit`.")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    fmt: Annotated[ReportFormat, typer.Option("--format", "-f")] = ReportFormat.html,
) -> None:
    """Re-render a saved JSON result as HTML, Markdown, JSON or text."""
    from quantproof.reports import load_result, render

    try:
        result = load_result(result_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _error(f"Cannot read {result_json}: {exc}")
        return
    text = render(result, fmt.value)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Report written to {output}", err=True)
    else:
        typer.echo(text)


@app.command()
def validate(
    data: Annotated[Path, typer.Argument(help="Price data (.csv/.parquet).")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    timestamp_column: Annotated[str | None, typer.Option("--timestamp-column")] = None,
    fail_on: Annotated[FailOn, typer.Option("--fail-on")] = FailOn.fail,
) -> None:
    """Run data-quality checks (QP-DATA-*) on a price file."""
    from quantproof.audit.models import AuditResult
    from quantproof.data.loaders import load_frame
    from quantproof.data.validation import validate_data
    from quantproof.reports.text import render_text

    try:
        cfg = _load_config(config, False, None)
        findings = validate_data(
            load_frame(data, timestamp_column=timestamp_column),
            cfg.data,
            timestamp_column=timestamp_column,
        )
    except QuantProofError as exc:
        _error(str(exc))
        return
    result = AuditResult.from_findings(findings)
    typer.echo(render_text(result, verbose=True, color=_color()))
    raise typer.Exit(_exit_code(result.status.value, fail_on))


@app.command()
def scan(
    path: Annotated[Path, typer.Argument(help="Python file or directory.")],
    fail_on: Annotated[FailOn, typer.Option("--fail-on")] = FailOn.fail,
    show_passes: Annotated[bool, typer.Option("--show-passes")] = False,
) -> None:
    """Static analysis only (QP001–QP015) on a file or directory tree."""
    from quantproof.audit.models import AuditResult
    from quantproof.audit.static import analyze_path
    from quantproof.reports.text import render_text

    if not path.exists():
        _error(f"{path} does not exist.")
    findings = analyze_path(path, include_passes=show_passes)
    result = AuditResult.from_findings(findings)
    n_files = 1 if path.is_file() else sum(1 for _ in path.rglob("*.py"))
    typer.echo(render_text(result, verbose=True, color=_color()))
    if not result.issues:
        typer.echo(f"No WARN or FAIL findings in {n_files} Python file(s) under {path}.")
    raise typer.Exit(_exit_code(result.status.value, fail_on))


@app.command()
def rules() -> None:
    """List every static and runtime rule identifier."""
    from quantproof.audit.static import list_rules

    for r in list_rules():
        typer.echo(f"{r['id']:<8} {r['title']}")


@app.command()
def statistics(
    returns: Annotated[Path, typer.Argument(help="Returns file (.csv/.parquet).")],
    column: Annotated[str | None, typer.Option("--column")] = None,
    periods: Annotated[float, typer.Option("--periods", help="Periods per year.")] = 252.0,
    trials: Annotated[
        int, typer.Option("--trials", help="Number of variants tried (for DSR).")
    ] = 1,
    benchmark_sharpe: Annotated[float, typer.Option("--benchmark-sharpe")] = 0.0,
    risk_free: Annotated[float, typer.Option("--risk-free", help="Annual risk-free rate.")] = 0.0,
    n_boot: Annotated[int, typer.Option("--bootstrap", help="Bootstrap resamples.")] = 1000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Sharpe, PSR, DSR, bootstrap CI and minimum track record length of a returns series."""
    from quantproof._utils import format_number, to_jsonable
    from quantproof.data.loaders import load_returns
    from quantproof.statistics import (
        bootstrap_sharpe,
        deflated_sharpe_ratio,
        minimum_track_record_length,
        probabilistic_sharpe_ratio,
        sharpe_summary,
    )

    try:
        r = load_returns(returns, column=column).dropna()
        s = sharpe_summary(r, periods_per_year=periods, risk_free_rate=risk_free)
        psr = probabilistic_sharpe_ratio(
            r, benchmark_sharpe=benchmark_sharpe, periods_per_year=periods, risk_free_rate=risk_free
        )
        dsr = deflated_sharpe_ratio(
            r, n_trials=trials, periods_per_year=periods, risk_free_rate=risk_free
        )
        boot = bootstrap_sharpe(r, periods_per_year=periods, n_boot=n_boot, seed=seed)
        mintrl = minimum_track_record_length(
            s.sharpe_per_period,
            s.skewness,
            s.kurtosis,
            benchmark_sharpe=benchmark_sharpe / periods**0.5,
        )
    except QuantProofError as exc:
        _error(str(exc))
        return
    out = {
        "sharpe": s.to_dict(),
        "psr": psr.to_dict(),
        "dsr": dsr.to_dict(),
        "bootstrap": boot.to_dict(),
        "min_track_record_length": mintrl,
    }
    if as_json:
        typer.echo(json.dumps(to_jsonable(out), indent=2))
        return
    typer.echo(f"Observations:                 {s.n_observations}")
    typer.echo(
        f"Sharpe (annualized):          {format_number(s.sharpe_annualized, 3)}  (s.e. {format_number(s.standard_error_annualized, 3)})"
    )
    typer.echo(
        f"Skewness / kurtosis:          {format_number(s.skewness, 2)} / {format_number(s.kurtosis, 2)}"
    )
    typer.echo(
        f"Probabilistic Sharpe Ratio:   {format_number(psr.psr, 4)}  (benchmark {benchmark_sharpe})"
    )
    typer.echo(
        f"Deflated Sharpe Ratio:        {format_number(dsr.dsr, 4)}  (trials {trials}, "
        f"E[max SR] {format_number(dsr.to_dict()['expected_max_sharpe_annualized'], 3)})"
    )
    typer.echo(
        f"Bootstrap {boot.confidence:.0%} CI:             [{format_number(boot.ci_lower, 3)}, {format_number(boot.ci_upper, 3)}]"
        f"  ({boot.method}, block {format_number(boot.block_length, 1)})"
    )
    typer.echo(f"Min. track record length:     {format_number(mintrl, 0)} observations")


@app.command("init-config")
def init_config(
    path: Annotated[Path, typer.Argument()] = Path("quantproof.yaml"),
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Write a configuration file with every default value."""
    import yaml

    from quantproof.config import AuditConfig

    if path.exists() and not force:
        _error(f"{path} exists; use --force to overwrite.")
    path.write_text(yaml.safe_dump(AuditConfig().to_dict(), sort_keys=False), encoding="utf-8")
    typer.echo(f"Wrote {path}")


@app.command("generate-data")
def generate_data(
    output: Annotated[Path, typer.Argument(help="Output .csv or .parquet path.")],
    n: Annotated[int, typer.Option("--n")] = 1500,
    seed: Annotated[int, typer.Option("--seed")] = 2026,
    ar1: Annotated[float, typer.Option("--ar1", help="First-order autocorrelation.")] = -0.15,
) -> None:
    """Write seeded synthetic OHLCV data (for trying QuantProof offline)."""
    from quantproof.data.synthetic import generate_prices

    df = generate_prices(n, seed=seed, ar1=ar1, start="2019-01-01")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        df.round(6).to_csv(output)
    elif output.suffix.lower() in {".parquet", ".pq"}:
        try:
            df.to_parquet(output)
        except ImportError:
            _error("Writing Parquet requires pyarrow: pip install 'quantproof[parquet]'")
    else:
        _error("Output must end in .csv or .parquet")
    typer.echo(f"Wrote {len(df)} rows to {output}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
