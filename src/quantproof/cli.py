"""Command-line interface.

Exit codes
----------
====  ==========================================================================
0     Completed; the overall status is PASS/INFO, or below the ``--fail-on`` level.
1     Completed; the overall status is WARN (and ``--fail-on`` is ``warn``).
2     Completed; the overall status is FAIL (and ``--fail-on`` is not ``never``).
3     Invalid input: bad option, unreadable file, data or strategy QuantProof cannot
      use, unknown rule id. Nothing was audited.
4     Internal error: a bug in QuantProof. Re-run with ``--debug`` for a traceback.
====  ==========================================================================

``--fail-on`` sets the lowest status that produces a non-zero code: ``warn`` (default:
WARN → 1, FAIL → 2), ``fail`` (WARN → 0, FAIL → 2) or ``never`` (always 0).
"""

from __future__ import annotations

import json
import sys
import traceback
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from typer.core import TyperGroup

from quantproof._version import __version__
from quantproof.errors import QuantProofError

EXIT_OK = 0
EXIT_WARN = 1
EXIT_FAIL = 2
EXIT_INVALID_INPUT = 3
EXIT_INTERNAL_ERROR = 4

_STATE: dict[str, bool] = {"debug": False}


def _is_click_error(exc: BaseException, name: str) -> bool:
    """Duck-typed check for click/typer exception classes (typer may vendor click)."""
    return any(cls.__name__ == name for cls in type(exc).__mro__)


class _QuantProofGroup(TyperGroup):
    """Maps every outcome to the documented exit codes and hides raw tracebacks."""

    def main(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["standalone_mode"] = False
        try:
            rv = super().main(*args, **kwargs)
        except typer.Abort:
            typer.echo("Aborted.", err=True)
            sys.exit(130)
        except QuantProofError as exc:
            typer.echo(f"Error: {exc}", err=True)
            sys.exit(EXIT_INVALID_INPUT)
        except Exception as exc:
            if _is_click_error(exc, "NoArgsIsHelpError"):
                exc.show()  # type: ignore[attr-defined]
                sys.exit(EXIT_OK)
            if _is_click_error(exc, "ClickException"):
                exc.show()  # type: ignore[attr-defined]
                sys.exit(EXIT_INVALID_INPUT)
            if _STATE["debug"]:
                traceback.print_exc()
            else:
                typer.echo(
                    f"Internal error: {type(exc).__name__}: {exc}\n"
                    "This is a bug in QuantProof. Re-run with `quantproof --debug ...` for a "
                    "traceback and report it at https://github.com/z33shxn/QuantProof/issues.",
                    err=True,
                )
            sys.exit(EXIT_INTERNAL_ERROR)
        sys.exit(rv if isinstance(rv, int) else EXIT_OK)


app = typer.Typer(
    name="quantproof",
    cls=_QuantProofGroup,
    help=(
        "QuantProof — audit a backtest for look-ahead bias, leakage, unrealistic execution "
        "and selection bias.\n\n"
        "Exit codes: 0 pass, 1 warn, 2 fail, 3 invalid input, 4 internal error."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
rules_app = typer.Typer(
    help="List rules, or show one rule's documentation (`quantproof rules show QP001`).",
    invoke_without_command=True,
)
app.add_typer(rules_app, name="rules")


class ReportFormat(str, Enum):
    html = "html"
    json = "json"
    markdown = "markdown"
    text = "text"


class FailOn(str, Enum):
    warn = "warn"
    fail = "fail"
    never = "never"


class Profile(str, Enum):
    quick = "quick"
    standard = "standard"
    strict = "strict"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"quantproof {__version__}")
        raise typer.Exit(EXIT_OK)


@app.callback()
def _main_options(
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Show Python tracebacks for internal errors."),
    ] = False,
    _version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Print the version."
        ),
    ] = False,
) -> None:
    """Global options."""
    _STATE["debug"] = debug


def _color() -> bool:
    return sys.stdout.isatty()


def _error(msg: str) -> None:
    typer.echo(f"Error: {msg}", err=True)
    raise typer.Exit(EXIT_INVALID_INPUT)


def _exit_code(status: str, fail_on: FailOn) -> int:
    if fail_on is FailOn.never:
        return EXIT_OK
    if status == "FAIL":
        return EXIT_FAIL
    if status == "WARN" and fail_on is FailOn.warn:
        return EXIT_WARN
    return EXIT_OK


def _load_config(
    path: Path | None, profile: Profile | None, seed: int | None, *, quick: bool = False
) -> Any:
    from quantproof.config import AuditConfig

    cfg = AuditConfig.from_file(path) if path else AuditConfig()
    if quick and profile not in (None, Profile.quick):
        _error("--quick is an alias for --profile quick; do not combine it with another profile.")
    if quick:
        cfg.profile = "quick"
    elif profile is not None:
        cfg.profile = profile.value
    if seed is not None:
        cfg.seed = seed
    return cfg


ProfileOpt = Annotated[
    Profile | None,
    typer.Option(
        "--profile",
        "-p",
        help=(
            "quick: smaller resampling sizes for fast iteration; standard: defaults; strict: "
            "larger resampling sizes and declared trial counts required."
        ),
    ),
]
FailOnOpt = Annotated[
    FailOn,
    typer.Option("--fail-on", help="Lowest status that gives a non-zero exit code."),
]


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
    profile: ProfileOpt = None,
    quick: Annotated[bool, typer.Option("--quick", help="Alias for --profile quick.")] = False,
    seed: Annotated[int | None, typer.Option("--seed", help="Override the random seed.")] = None,
    timestamp_column: Annotated[
        str | None, typer.Option("--timestamp-column", help="Timestamp column name.")
    ] = None,
    symbol_column: Annotated[
        str | None,
        typer.Option("--symbol-column", help="Symbol column of multi-asset (long) data."),
    ] = None,
    fail_on: FailOnOpt = FailOn.warn,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Do not print the summary.")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show every finding, including passes.")
    ] = False,
) -> None:
    """Audit a strategy (--strategy with --data) or pre-computed returns (--returns/--trials).

    Examples:

      quantproof audit -s strategy.py -d prices.csv -o report.html

      quantproof audit --returns returns.csv --trials trials.csv --profile strict
    """
    from quantproof.data.loaders import load_frame, load_returns, prepare_prices
    from quantproof.engine import audit
    from quantproof.reports import render, write_report
    from quantproof.reports.text import render_text

    cfg = _load_config(config, profile, seed, quick=quick)
    try:
        if strategy is not None:
            if data is None:
                typer.echo("Note: no --data given; only static analysis will run.", err=True)
            result = audit(
                strategy=strategy,
                data=data,
                config=cfg,
                benchmark=benchmark,
                timestamp_column=timestamp_column,
                symbol_column=symbol_column,
            )
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
    symbol_column: Annotated[
        str | None, typer.Option("--symbol-column", help="Symbol column of multi-asset data.")
    ] = None,
    fail_on: FailOnOpt = FailOn.warn,
) -> None:
    """Run data-quality checks (QP-DATA-*) on a price file (single asset or long panel)."""
    from quantproof.data.loaders import load_frame
    from quantproof.data.validation import validate_data
    from quantproof.reports.text import render_text
    from quantproof.results import AuditResult

    try:
        cfg = _load_config(config, None, None)
        findings = validate_data(
            load_frame(data, timestamp_column=timestamp_column),
            cfg.data,
            timestamp_column=timestamp_column,
            symbol_column=symbol_column,
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
    fail_on: FailOnOpt = FailOn.warn,
    show_passes: Annotated[bool, typer.Option("--show-passes")] = False,
) -> None:
    """Static analysis only (QP001–QP015) on a file or directory tree."""
    from quantproof.analyzers.static import analyze_path
    from quantproof.reports.text import render_text
    from quantproof.results import AuditResult

    if not path.exists():
        _error(f"{path} does not exist.")
    findings = analyze_path(path, include_passes=show_passes)
    result = AuditResult.from_findings(findings)
    n_files = 1 if path.is_file() else sum(1 for _ in path.rglob("*.py"))
    typer.echo(render_text(result, verbose=True, color=_color()))
    if not result.issues:
        typer.echo(f"No WARN or FAIL findings in {n_files} Python file(s) under {path}.")
    raise typer.Exit(_exit_code(result.status.value, fail_on))


@rules_app.callback()
def rules_list(
    ctx: typer.Context,
    category: Annotated[
        str | None, typer.Option("--category", help="Only rules in this category.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List every rule (data, static, runtime, execution, statistics …)."""
    if ctx.invoked_subcommand is not None:
        return
    from quantproof.rules import list_rules

    specs = list_rules(category)
    if category and not specs:
        _error(f"No rules in category {category!r}.")
    if as_json:
        typer.echo(json.dumps([r.to_dict() for r in specs], indent=2, ensure_ascii=False))
        return
    for r in specs:
        typer.echo(f"{r.id:<14} {r.max_severity.value:<5} {r.category:<12} {r.name}")


@rules_app.command("show")
def rules_show(
    rule_id: Annotated[str, typer.Argument(help="Rule id, e.g. QP001 or QP-STAT-003.")],
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Show one rule's documentation: severity policy, rationale, limitations, example."""
    from quantproof.rules import get_rule

    spec = get_rule(rule_id)
    if as_json:
        typer.echo(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False))
        return
    fields = [
        ("Name", spec.name),
        ("Category", spec.category),
        ("Analysis", spec.analysis),
        ("Max severity", spec.max_severity.value),
        ("Severity policy", spec.severity_policy),
        ("What it checks", spec.description),
        ("Why it matters", spec.rationale),
        ("Potential impact", spec.impact),
        ("How to investigate", spec.investigate),
        ("Remediation", spec.remediation),
        ("Limitations", spec.limitations),
    ]
    typer.echo(spec.id)
    for label, value in fields:
        if value:
            typer.echo(f"  {label + ':':<20}{value}")
    if spec.example:
        typer.echo("  Example:")
        for line in spec.example.splitlines():
            typer.echo(f"    {line}")


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
    """Console-script entry point (exit codes documented in the module docstring)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
