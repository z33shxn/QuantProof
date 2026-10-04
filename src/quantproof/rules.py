"""Machine-readable registry of every rule QuantProof can report.

The registry is the single source of truth for rule metadata: identifiers, names,
categories, the maximum severity a rule can emit and the policy that decides it,
what the rule checks, why it matters, the potential impact on results, how to
investigate, how to fix, known limitations, and an example.

>>> from quantproof.rules import get_rule, list_rules
>>> get_rule("QP001").name
'Negative temporal shift'
>>> len([r for r in list_rules() if r.analysis == "static"])
16

``confidence`` on a finding is *analysis confidence* — how sure the analyzer is that
the pattern it matched means what the rule says. It is not a statistical probability.
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass
from typing import Any

from quantproof.errors import QuantProofInputError
from quantproof.severity import Severity


@dataclass(frozen=True)
class RuleSpec:
    """Metadata for one rule."""

    id: str
    name: str
    category: str
    analysis: str  # "data" | "static" | "runtime" | "statistical" | "meta"
    max_severity: Severity
    severity_policy: str
    description: str
    rationale: str
    impact: str
    investigate: str
    remediation: str
    limitations: str
    example: str = ""
    pass_title: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["max_severity"] = self.max_severity.value
        return d


_RULES: list[RuleSpec] = []


def _add(**kw: Any) -> None:
    _RULES.append(RuleSpec(**kw))


F, W, INFO_ = Severity.FAIL, Severity.WARN, Severity.INFO
_LOOKAHEAD_IMPACT = (
    "Backtest performance may be overstated, potentially dramatically: decisions are "
    "credited with information that only became available later."
)
_LIVE = "A live trading system could not know that information at decision time."

# --------------------------------------------------------------------------- data
_DATA = [
    (
        "QP-DATA-000",
        "Data preparation actions",
        INFO_,
        "INFO only; records every fix applied before analysis.",
        "Records rows dropped, re-sorted or de-duplicated before analysis.",
        "Nothing is changed silently; the audit runs on the prepared data.",
        "Results describe the prepared data, not the raw file.",
        "Compare the action list with the raw file.",
        "Fix the source data.",
        "Purely descriptive.",
    ),
    (
        "QP-DATA-001",
        "Invalid timestamps",
        W,
        "WARN when any timestamp cannot be parsed.",
        "Timestamps that cannot be parsed (the rows are dropped).",
        "Rows without a valid time cannot be placed in sequence.",
        "Dropped rows shorten the sample and may hide gaps.",
        "Look at the listed examples in the raw file.",
        "Fix malformed timestamps at the source.",
        "Parsing uses pandas' mixed-format parser; ambiguous day/month orders are not detected.",
    ),
    (
        "QP-DATA-002",
        "Non-monotonic timestamps",
        W,
        "WARN when timestamps go backwards (rows are sorted).",
        "Timestamps that are not in increasing order.",
        "shift/rolling operations on unsorted data reference the wrong neighbours.",
        "Signals computed before sorting may contain look-ahead.",
        "Check how the file was assembled (merges, appends).",
        "Sort at ingestion.",
        "",
    ),
    (
        "QP-DATA-003",
        "Missing price or OHLC fields",
        F,
        "FAIL without any price column, or without OHLC when require_ohlc; INFO for partial OHLC.",
        "Presence of a price column and of open/high/low/close.",
        "Returns need prices; next-open fills and bar-consistency checks need OHLC.",
        "Some execution scenarios cannot be simulated.",
        "List the columns of the file.",
        "Provide close (and ideally OHLC) columns.",
        "",
    ),
    (
        "QP-DATA-004",
        "Duplicate timestamps",
        W,
        "WARN (INFO if allow_duplicate_timestamps); in panels, duplicates of (timestamp, symbol).",
        "Repeated timestamps (per symbol for multi-asset data).",
        "Duplicates distort returns, rolling features, signal timing and turnover.",
        "The last record per timestamp is kept; earlier ones are ignored.",
        "Inspect the affected range.",
        "De-duplicate at the source.",
        "Legitimate multi-record data (several trades per timestamp) must be aggregated first.",
    ),
    (
        "QP-DATA-005",
        "Duplicated observations",
        W,
        "WARN when entire rows repeat.",
        "Rows identical in timestamp and every value.",
        "Usually a faulty merge or double ingestion.",
        "Double-counted bars.",
        "Check the ingestion pipeline.",
        "Remove duplicate rows.",
        "",
    ),
    (
        "QP-DATA-006",
        "Impossible OHLC relationships",
        F,
        "FAIL when high/low do not bound open/close.",
        "low ≤ min(open, close) ≤ max(open, close) ≤ high for every complete bar.",
        "Impossible bars indicate corrupted data.",
        "Fills, stops and range features are unreliable.",
        "Inspect the listed bars.",
        "Repair or drop corrupted bars.",
        "",
    ),
    (
        "QP-DATA-007",
        "Non-positive prices",
        F,
        "FAIL on zero or negative prices.",
        "Zero or negative values in price columns.",
        "Percentage returns are undefined.",
        "Returns become infinite or meaningless.",
        "Inspect the affected range.",
        "Fix the data, or model P&L in price differences for instruments that can trade below zero.",
        "Instruments such as some spreads or energy futures can legitimately trade at or below zero.",
    ),
    (
        "QP-DATA-008",
        "Missing values",
        W,
        "WARN when any column has missing values.",
        "Missing values per column.",
        "Implicit fill choices (especially back-fill) can leak information.",
        "NaNs propagate through rolling features and returns.",
        "See the per-column counts.",
        "Decide explicitly how gaps are treated; never back-fill prices.",
        "",
    ),
    (
        "QP-DATA-009",
        "Unexplained gaps",
        W,
        "WARN when gaps exceed max_gap_multiple × median spacing.",
        "Gaps between consecutive timestamps (per symbol).",
        "A 'bar' then spans a long period; bar-count lags change meaning.",
        "Volatility and lag assumptions are distorted.",
        "Check whether gaps are market closures.",
        "Fill the source data or raise the threshold.",
        "Intraday gaps across calendar days are treated as session breaks; exchange calendars are not modelled.",
    ),
    (
        "QP-DATA-010",
        "Timezone inconsistency",
        W,
        "WARN for mixed tz-aware/naive timestamps; INFO for several UTC offsets (e.g. DST).",
        "Mixtures of timezone conventions.",
        "Mixed conventions can shift bars by hours.",
        "Events may be misordered relative to signals.",
        "Inspect the raw timestamp strings.",
        "Store timestamps in UTC with explicit offsets.",
        "Naive timestamps are assumed consistent.",
    ),
    (
        "QP-DATA-011",
        "Stale (forward-filled) prices",
        W,
        "WARN on runs of ≥ max_stale_run identical closes.",
        "Runs of identical consecutive closes.",
        "Forward-filled prices create zero returns and fake fills.",
        "Understated volatility; fills at prices that never traded.",
        "Inspect the longest run.",
        "Mark non-trading periods explicitly.",
        "Illiquid instruments can legitimately repeat prices.",
    ),
    (
        "QP-DATA-012",
        "Infinite values",
        F,
        "FAIL on any infinite numeric value.",
        "Infinite numeric values.",
        "They poison every downstream statistic.",
        "Statistics become undefined.",
        "Find the computation that produced them.",
        "Fix the upstream division by zero.",
        "",
    ),
    (
        "QP-DATA-013",
        "Extreme single-period returns",
        W,
        "WARN when |return| exceeds extreme_return_threshold.",
        "Single-period moves larger than the threshold (default 50 %).",
        "Often unadjusted splits, bad ticks or symbol changes.",
        "A few bars can dominate P&L.",
        "Check corporate actions around the listed dates.",
        "Adjust or remove bad ticks.",
        "Genuine extreme moves exist (crashes, small caps).",
    ),
    (
        "QP-DATA-014",
        "Empty or insufficient data",
        F,
        "FAIL when no usable observations remain.",
        "Whether any observations are available after parsing.",
        "Nothing can be validated or audited.",
        "Every downstream result would be undefined.",
        "Check the file and the timestamp column.",
        "Provide data.",
        "",
    ),
    (
        "QP-DATA-015",
        "Unsynchronized symbols",
        W,
        "WARN when symbols in a panel cover different timestamp sets; INFO for different start/end only.",
        "Whether all symbols share the same timestamps.",
        "Cross-sectional signals on unsynchronized data compare stale and fresh prices.",
        "Cross-sectional ranks may use information from different moments.",
        "See per-symbol coverage in the evidence.",
        "Align symbols on a common calendar explicitly.",
        "Listings and delistings legitimately change coverage.",
    ),
]
for rid, name, sev, policy, desc, why, impact, inv, fix, lim in _DATA:
    _add(
        id=rid,
        name=name,
        category="data",
        analysis="data",
        max_severity=sev,
        severity_policy=policy,
        description=desc,
        rationale=why,
        impact=impact,
        investigate=inv,
        remediation=fix,
        limitations=lim,
    )

# ------------------------------------------------------------------------- static
_add(
    id="QP000",
    name="Source cannot be parsed",
    category="static",
    analysis="static",
    max_severity=F,
    severity_policy="FAIL when the file is not valid Python.",
    description="The strategy source failed to parse.",
    rationale="Unparseable code cannot be audited.",
    impact="No static evidence is available.",
    investigate="Read the syntax error location.",
    remediation="Fix the syntax error.",
    limitations="",
)
_STATIC: list[dict[str, Any]] = [
    {
        "id": "QP001",
        "name": "Negative temporal shift",
        "severity_policy": "FAIL (high confidence) when traced to a decision or model input; INFO when used "
        "only to build a label; WARN when its use cannot be determined.",
        "description": "shift/diff/pct_change with a negative period, including through simple helpers and aliases.",
        "rationale": "shift(-k) moves future observations into the current row. " + _LIVE,
        "impact": _LOOKAHEAD_IMPACT,
        "investigate": "Follow the flagged value to where it is used; check signal and feature construction.",
        "remediation": "Use negative shifts only for labels that never feed features or signals.",
        "limitations": "Intra-module data flow only; values passed through containers, other modules or "
        "dynamic attributes are not traced.",
        "example": 'df["future"] = df["close"].shift(-1)',
        "pass_title": "No negative temporal shift detected",
    },
    {
        "id": "QP002",
        "name": "Centered rolling window",
        "severity_policy": "FAIL when traced to a decision or model input; WARN otherwise (e.g. plotting).",
        "description": "rolling(..., center=True).",
        "rationale": "A centered window at t averages about window/2 future observations.",
        "impact": _LOOKAHEAD_IMPACT,
        "investigate": "Check whether the smoothed series feeds signals.",
        "remediation": "Use trailing windows for anything that feeds decisions.",
        "limitations": "Centered windows for descriptive plots are legitimate; they are reported as WARN.",
        "example": 'df["close"].rolling(20, center=True).mean()',
        "pass_title": "No centered rolling window detected",
    },
    {
        "id": "QP003",
        "name": "Future-oriented construction",
        "severity_policy": "FAIL when traced to a decision or model input; WARN otherwise.",
        "description": "x[i + k] inside time loops, np.roll, and future-named model inputs.",
        "rationale": "These read beyond the current bar.",
        "impact": _LOOKAHEAD_IMPACT,
        "investigate": "Check whether the value is used for fills/labels (fine) or decisions (not).",
        "remediation": "Index only bars ≤ i for the decision at bar i; use shift instead of np.roll.",
        "limitations": "Name-based detection of future-named inputs has low confidence.",
        "example": "signal = 1 if close[i + 1] > close[i] else 0",
        "pass_title": "No future-oriented construction detected",
    },
    {
        "id": "QP004",
        "name": "Random train/test split",
        "max_severity": W,
        "severity_policy": "WARN; context-dependent (valid for i.i.d. data).",
        "description": "train_test_split without shuffle=False.",
        "rationale": "Shuffling time-ordered rows puts the future in the training set.",
        "impact": "Out-of-sample estimates are biased upward.",
        "investigate": "Check whether rows are ordered in time.",
        "remediation": "Use shuffle=False, walk-forward, PurgedKFold or CPCV.",
        "limitations": "Cannot tell whether the data are i.i.d.; legitimate for i.i.d. samples.",
        "example": "train_test_split(X, y, test_size=0.2)",
        "pass_title": "No shuffled train/test split detected",
    },
    {
        "id": "QP005",
        "name": "Randomized or non-temporal cross-validation",
        "max_severity": W,
        "severity_policy": "WARN for shuffling splitters; INFO for unshuffled K-fold and default CV.",
        "description": "KFold(shuffle=True), ShuffleSplit, StratifiedKFold, … and default K-fold in searches.",
        "rationale": "Models learn from the future and from overlapping labels.",
        "impact": "Cross-validated scores overstate real-time performance.",
        "investigate": "Check the splitter passed to cross-validation.",
        "remediation": "Use TimeSeriesSplit, PurgedKFold or CPCV.",
        "limitations": "Context-dependent; valid for i.i.d. samples.",
        "example": "KFold(5, shuffle=True)",
        "pass_title": "No randomized cross-validation detected",
    },
    {
        "id": "QP006",
        "name": "Preprocessing fitted before the split",
        "severity_policy": "FAIL (medium confidence) when the fitted data or its transform is later split; "
        "WARN when the fit precedes a split in the same scope but the link is not visible.",
        "description": "Scalers, imputers, PCA or selectors fitted on data that is later split.",
        "rationale": "Test-period statistics shape the training data.",
        "impact": "Out-of-sample scores are contaminated.",
        "investigate": "Check what data the transformer is fitted on.",
        "remediation": "Split first; fit on the training fold (e.g. sklearn Pipeline in temporal CV).",
        "limitations": "Detects scikit-learn style transformers bound to names or constructed inline.",
        "example": "X_s = StandardScaler().fit_transform(X)\ntrain_test_split(X_s, y)",
        "pass_title": "No fit-before-split leakage detected",
    },
    {
        "id": "QP007",
        "name": "Full-sample normalization",
        "max_severity": W,
        "severity_policy": "WARN (medium); the runtime causality test provides the definitive evidence.",
        "description": "(x - x.mean()) / x.std(), zscore/scale, transformers fitted with no split.",
        "rationale": "Statistics of the whole sample (including the future) define values at t.",
        "impact": _LOOKAHEAD_IMPACT,
        "investigate": "Check whether the series is time-indexed.",
        "remediation": "Use rolling/expanding statistics or fit on the training window.",
        "limitations": "Cross-sectional normalization within a date is legitimate; groupby/apply(axis=1) is "
        "recognized, other forms are not.",
        "example": "(x - x.mean()) / x.std()",
        "pass_title": "No full-sample normalization detected",
    },
    {
        "id": "QP008",
        "name": "Target leakage",
        "severity_policy": "FAIL when the target column (or a direct copy) is in the feature matrix.",
        "description": "Target included in features, not dropped, copied under another name, or fit(X, X).",
        "rationale": "The model can read the answer.",
        "impact": "Model scores become meaningless.",
        "investigate": "Compare the feature list with the target definition.",
        "remediation": "Exclude the target and anything derived from it.",
        "limitations": "Copies are traced through simple column assignments only.",
        "example": 'X = df[["ret", "target"]]\nmodel.fit(X, df["target"])',
        "pass_title": "No direct target leakage detected",
    },
    {
        "id": "QP009",
        "name": "Same-bar execution assumption",
        "max_severity": W,
        "severity_policy": "WARN when signal_lag=0 is declared or close-to-close same-bar fills are implied "
        "without a documented assumption; INFO when documented.",
        "description": "Signals from a bar's close filled at that same close.",
        "rationale": "Requires zero latency and an exact closing fill.",
        "impact": "Short-horizon edges can disappear with a realistic delay.",
        "investigate": "Check the execution timing assumed by the backtest.",
        "remediation": "Declare EXECUTION and test with a one-bar lag or next-open fills.",
        "limitations": "Market-on-close orders on signals computed shortly before the close can be valid.",
        "example": 'EXECUTION = {"signal_lag": 0}',
        "pass_title": "No undocumented same-bar execution detected",
    },
    {
        "id": "QP010",
        "name": "Missing execution lag",
        "severity_policy": "FAIL when both the signal and the returns are close-based; WARN when execution "
        "semantics cannot be determined; nothing when the signal uses only the open and returns are "
        "open-to-close.",
        "description": "An un-lagged signal multiplied by same-period returns.",
        "rationale": "The strategy is credited with a move that happened before the signal existed.",
        "impact": _LOOKAHEAD_IMPACT,
        "investigate": "Check the alignment of positions and returns.",
        "remediation": "Lag positions: returns * signal.shift(1).",
        "limitations": "Provenance is traced through simple assignments in the same file.",
        "example": 'df["strategy"] = df["signal"] * df["returns"]',
        "pass_title": "Signals are lagged before being multiplied by returns",
    },
    {
        "id": "QP011",
        "name": "Missing transaction costs",
        "max_severity": W,
        "severity_policy": "WARN when strategy returns are computed without any cost term, or EXECUTION "
        "declares only zero costs.",
        "description": "Backtest returns without commissions, spread, slippage or impact.",
        "rationale": "Gross returns overstate what can be earned.",
        "impact": "High-turnover strategies can flip from profitable to unprofitable.",
        "investigate": "Look for cost terms in the P&L calculation.",
        "remediation": "Subtract turnover-proportional costs or declare them in EXECUTION.",
        "limitations": "Detects cost terms by name.",
        "example": "pnl = position.shift(1) * returns",
        "pass_title": "Transaction costs referenced or not applicable",
    },
    {
        "id": "QP012",
        "name": "Future-derived model features",
        "severity_policy": "FAIL (high) when future-derived data reaches fit/predict/transform inputs.",
        "description": "Future values used as model inputs.",
        "rationale": _LIVE,
        "impact": _LOOKAHEAD_IMPACT,
        "investigate": "Follow the origin lines listed in the evidence.",
        "remediation": "Build features only from information at or before each timestamp.",
        "limitations": "Intra-module data flow only.",
        "example": 'X = df[["ret_1", "fwd_ret"]]\nmodel.fit(X, y)',
        "pass_title": "No future-derived model features detected",
    },
    {
        "id": "QP013",
        "name": "Large parameter search",
        "max_severity": W,
        "severity_policy": "WARN above max_trials_threshold (default 100); INFO otherwise with the count.",
        "description": "Grid/random searches, nested parameter loops, itertools.product, PARAM_GRID.",
        "rationale": "The best of many backtests is biased upward even without skill.",
        "impact": "Headline Sharpe overstates expected performance.",
        "investigate": "Count every configuration that was tried.",
        "remediation": "Report the trial count (statistics.trials) so the DSR accounts for it.",
        "limitations": "Counts only searches visible in this file with literal sizes.",
        "example": 'GridSearchCV(model, {"a": range(50), "b": range(50)})',
        "pass_title": "No large hyper-parameter search detected",
    },
    {
        "id": "QP014",
        "name": "No explicit out-of-sample evaluation",
        "max_severity": W,
        "severity_policy": "WARN when models are fitted or parameters searched with no split/CV construct.",
        "description": "Fitting or searching with no train/test separation visible in the file.",
        "rationale": "In-sample fit measures memorization, not generalization.",
        "impact": "Reported performance may not generalize.",
        "investigate": "Check whether evaluation happens elsewhere.",
        "remediation": "Hold out a test period or use walk-forward/purged CV.",
        "limitations": "Evaluation in another module is invisible; a declared PARAM_GRID is evaluated by "
        "QuantProof itself.",
        "example": "model.fit(X, y)\nmodel.score(X, y)",
        "pass_title": "Out-of-sample evaluation present or not applicable",
    },
    {
        "id": "QP015",
        "name": "Non-causal transformation",
        "severity_policy": "FAIL when traced to a decision or model input; WARN otherwise.",
        "description": "filtfilt, Savitzky-Golay, Gaussian smoothing, HP filter, seasonal decomposition, FFT, "
        "linear interpolate, backward fill.",
        "rationale": "These use observations after t to compute the value at t.",
        "impact": _LOOKAHEAD_IMPACT,
        "investigate": "Check whether the transformed series feeds decisions.",
        "remediation": "Use one-sided filters (lfilter, ewm) and forward-fill only.",
        "limitations": "Library-specific names; custom two-sided filters are not recognized.",
        "example": 'df["close"].bfill()',
        "pass_title": "No non-causal transformation detected",
    },
]
for spec in _STATIC:
    spec.setdefault("max_severity", F)
    _add(category="static", analysis="static", **spec)

# ------------------------------------------------------------------------ runtime
_RUNTIME = [
    (
        "QP-CAUSAL-001",
        "Future perturbation changed past decisions",
        "causality",
        F,
        "FAIL when any decision at or before t changes after data strictly after t is perturbed.",
        "Re-runs the strategy with data after a decision time modified and compares earlier outputs.",
        _LIVE,
        _LOOKAHEAD_IMPACT,
        "Inspect the listed decision times, original vs perturbed values, and the static findings.",
        "Rebuild the signal using data available no later than the decision timestamp.",
        "Evidence for tested timestamps and schemes only; cannot see leakage embedded in the data itself.",
    ),
    (
        "QP-CAUSAL-002",
        "Non-deterministic strategy",
        "causality",
        W,
        "WARN when two runs on identical data differ (test inconclusive).",
        "Determinism of the strategy output.",
        "Irreproducible output cannot be tested for causality.",
        "Results cannot be reproduced.",
        "Look for unseeded randomness.",
        "Seed every random generator.",
        "",
    ),
    (
        "QP-CAUSAL-003",
        "Strategy errors on perturbed data",
        "causality",
        INFO_,
        "INFO; the affected trials are not evaluated.",
        "Exceptions raised on perturbed inputs.",
        "Untested trials reduce the evidence.",
        "Weaker causality evidence.",
        "See the error messages.",
        "Make the strategy robust to unusual but valid prices.",
        "",
    ),
    (
        "QP-CAUSAL-004",
        "Output insensitive to data",
        "causality",
        INFO_,
        "INFO when perturbing all data never changes the output.",
        "Whether the output depends on the input.",
        "The causality test is uninformative for constant strategies.",
        "None.",
        "Check the strategy logic.",
        "",
        "",
    ),
]
for rid, name, cat, sev, policy, desc, why, impact, inv, fix, lim in _RUNTIME:
    _add(
        id=rid,
        name=name,
        category=cat,
        analysis="runtime",
        max_severity=sev,
        severity_policy=policy,
        description=desc,
        rationale=why,
        impact=impact,
        investigate=inv,
        remediation=fix,
        limitations=lim,
    )

_VALUES = [
    (
        "QP-LEAK-001",
        "Feature replicates the target",
        "leakage",
        F,
        "FAIL when |corr(feature, target)| ≥ 0.98.",
        "Value-level comparison of features and target.",
        "The model can read the answer.",
        "Model scores are meaningless.",
        "See the listed features.",
        "Remove target-derived features.",
        "Only near-exact copies are detected.",
    ),
    (
        "QP-LEAK-002",
        "Feature replicates a future return",
        "leakage",
        F,
        "FAIL when |corr(feature, r[t+h])| ≥ 0.98 for some 1 ≤ h ≤ 5.",
        "Value-level comparison of features with future returns.",
        _LIVE,
        _LOOKAHEAD_IMPACT,
        "See the feature and lead in the evidence.",
        "Rebuild the feature from data at or before t.",
        "Only near-exact copies are detected.",
    ),
    (
        "QP-LEAK-003",
        "Implausible directional accuracy",
        "leakage",
        W,
        "WARN when the hit rate vs the next return is above 75 % or below 25 % with two-sided binomial p < 1e-6 (≥ 50 bars).",
        "Pooled directional accuracy of signals against next-bar returns.",
        "Near-perfect accuracy on market data almost always indicates look-ahead.",
        _LOOKAHEAD_IMPACT,
        "Check signal inputs and alignment.",
        "Remove future inputs.",
        "A heuristic plausibility bound, not a proof.",
    ),
    (
        "QP-EXEC-001",
        "Performance depends on unrealistic execution",
        "execution",
        F,
        "FAIL when declared assumptions are optimistic (lag 0, zero costs, or none), the headline Sharpe "
        "≥ 0.5 and < 25 % survives the audit assumptions; WARN for declared lag 0 that survives.",
        "Compares naive, declared and realistic execution scenarios.",
        "Zero-latency closing fills and free trading are not achievable.",
        "Reported performance may be an artefact of execution assumptions.",
        "Read the lag- and cost-sensitivity tables.",
        "Evaluate with lagged fills and realistic costs.",
        "Fills at bar prices; no matching engine.",
    ),
    (
        "QP-EXEC-002",
        "Transaction-cost model missing",
        "execution",
        W,
        "WARN when no non-zero cost is declared; INFO when declared costs are below the auditor's.",
        "Whether the strategy declares cost assumptions.",
        "Gross results overstate earnings.",
        "Net performance may be much lower.",
        "See declared vs audit costs.",
        "Declare costs in EXECUTION.",
        "",
    ),
    (
        "QP-EXEC-003",
        "Edge fragile to transaction costs",
        "execution",
        W,
        "WARN when net Sharpe ≤ 0 while gross > 0, or the break-even cost is < 2× the assumed cost.",
        "Break-even cost and cost-multiplier sensitivity.",
        "Small cost errors would erase the return.",
        "The strategy may be unprofitable after real costs.",
        "Read the cost-sensitivity table.",
        "Reduce turnover or verify costs with real fills.",
        "Cost models are approximations.",
    ),
    (
        "QP-EXEC-004",
        "Excessive turnover relative to edge",
        "execution",
        W,
        "WARN when costs consume ≥ 50 % of the gross return; INFO otherwise.",
        "Turnover and cost drag.",
        "High turnover multiplies cost estimation error.",
        "Net returns are very sensitive to costs.",
        "See annualized turnover and cost drag.",
        "Add trading buffers or reduce rebalancing.",
        "",
    ),
    (
        "QP-EXEC-005",
        "Trades before or at their signal",
        "execution",
        F,
        "FAIL when a trade executes before its signal; WARN when > 50 % execute with zero latency.",
        "Signal vs execution timestamps of a trade ledger.",
        "Execution before a signal is look-ahead.",
        _LOOKAHEAD_IMPACT,
        "See the latency statistics.",
        "Record realistic execution timestamps.",
        "Requires signal_time and execution_time columns.",
    ),
    (
        "QP-EXEC-006",
        "Execution semantics not testable from bars",
        "execution",
        W,
        "WARN when declared fills are intrabar or event-driven.",
        "Execution styles that bar data cannot validate.",
        "QuantProof cannot verify fills that happen inside a bar.",
        "Execution results rely on unverified assumptions.",
        "Validate fills with tick or order data.",
        "Provide a trade ledger with timestamps, or declare a bar-level fill model.",
        "",
    ),
    (
        "QP-STAT-001",
        "Small sample",
        "statistics",
        W,
        "WARN below min_observations.",
        "Number of return observations.",
        "Short samples give very uncertain estimates.",
        "Sharpe confidence intervals are wide.",
        "See the bootstrap interval.",
        "Extend the sample.",
        "",
    ),
    (
        "QP-STAT-002",
        "Probabilistic Sharpe Ratio",
        "statistics",
        W,
        "WARN when PSR < confidence.",
        "Probability that the true Sharpe exceeds the benchmark (Bailey & López de Prado 2012).",
        "A positive Sharpe can arise by chance.",
        "Performance may be luck.",
        "See skew, kurtosis, MinTRL.",
        "Collect more out-of-sample data.",
        "Assumes i.i.d. returns and asymptotic normality.",
    ),
    (
        "QP-STAT-003",
        "Deflated Sharpe Ratio",
        "statistics",
        W,
        "WARN when DSR (declared trials) < confidence.",
        "PSR against the expected maximum Sharpe of N skill-less trials (Bailey & López de Prado 2014).",
        "Selection from many trials inflates the best Sharpe.",
        "Performance may be selection bias.",
        "See declared vs effective trials and the hurdle.",
        "Report all trials; reduce the search.",
        "Only declared trials are corrected for; trials are assumed independent.",
    ),
    (
        "QP-STAT-004",
        "Suspiciously strong statistics",
        "statistics",
        W,
        "WARN when any scenario's annualized Sharpe exceeds max_plausible_sharpe.",
        "Plausibility of Sharpe ratios.",
        "Very high Sharpe ratios usually indicate a bug.",
        "Results may be artefacts.",
        "Check look-ahead and execution findings.",
        "Treat as a bug until proven otherwise.",
        "High-frequency strategies can have high Sharpe ratios.",
    ),
    (
        "QP-STAT-005",
        "Number of trials not declared",
        "statistics",
        W,
        "INFO (WARN in the strict profile) when no trials are declared.",
        "Whether selection bias can be measured.",
        "DSR then equals PSR.",
        "Selection bias is unmeasured.",
        "Count the variants tried.",
        "Set statistics.trials.",
        "",
    ),
    (
        "QP-STAT-006",
        "Non-normal returns",
        "statistics",
        INFO_,
        "INFO when |skew| > 1 or excess kurtosis > 3.",
        "Skewness and kurtosis.",
        "Normal-theory intervals are unreliable.",
        "Tail risk is understated.",
        "See the moments.",
        "Use PSR/DSR and bootstrap intervals.",
        "",
    ),
    (
        "QP-STAT-007",
        "Serially correlated returns",
        "statistics",
        INFO_,
        "INFO when |ρ1| > 2/√n.",
        "Lag-1 autocorrelation.",
        "√T annualization is biased.",
        "Annual Sharpe is misstated.",
        "See the Lo-adjusted Sharpe.",
        "Use the adjusted Sharpe.",
        "Uses 10 lags.",
    ),
    (
        "QP-VAL-001",
        "Out-of-sample evidence",
        "validation",
        W,
        "WARN when OOS Sharpe ≤ 0, fewer than half the folds are positive, OOS < 50 % of a positive IS "
        "Sharpe, or OOS observations < min_observations.",
        "Walk-forward out-of-sample performance of the selection procedure.",
        "OOS performance of the full procedure is the honest estimate.",
        "Expected performance may be lower.",
        "See per-fold IS/OOS Sharpe and dispersion.",
        "Report OOS results as the headline.",
        "A single walk-forward path is noisy.",
    ),
    (
        "QP-VAL-002",
        "Probability of Backtest Overfitting",
        "validation",
        W,
        "WARN when PBO ≥ 0.5.",
        "CSCV estimate of how often the IS winner ranks below the OOS median (Bailey et al. 2017).",
        "The selection procedure may be fitting noise.",
        "Selected parameters may underperform.",
        "See the logit distribution and degradation.",
        "Simplify the search.",
        "Measures this sample's selection procedure.",
    ),
    (
        "QP-VAL-003",
        "CPCV path performance",
        "validation",
        W,
        "WARN when the median path Sharpe ≤ 0.",
        "Distribution of OOS Sharpe over CPCV paths (López de Prado 2018).",
        "One split can be lucky.",
        "Expected performance may be lower.",
        "See path Sharpe ratios.",
        "Prefer strategies with mostly positive paths.",
        "Paths share data and are not independent.",
    ),
    (
        "QP-VAL-004",
        "Data-snooping test",
        "validation",
        W,
        "WARN when Hansen's SPA p-value > 1 − confidence.",
        "White's Reality Check and Hansen's SPA over all configurations vs a zero benchmark.",
        "The best of many may be indistinguishable from luck.",
        "Edge may be selection bias.",
        "See both p-values.",
        "Treat as unproven unless the test rejects.",
        "Valid only if every tried configuration is included.",
    ),
    (
        "QP-REGIME-001",
        "Regime-dependent performance",
        "regime",
        W,
        "WARN when a regime with sufficient data has negative Sharpe while the overall Sharpe is positive.",
        "Performance by volatility, drawdown and trend regime.",
        "Results may depend on the regime mix.",
        "Performance may disappear when regimes change.",
        "See the regime tables.",
        "Size for adverse regimes.",
        "Simple regime definitions; few switches give noisy estimates.",
    ),
    (
        "QP-SENS-001",
        "Fragile parameter optimum",
        "sensitivity",
        W,
        "WARN when the plateau ratio < fragile_ratio; INFO for moderate; PASS for robust.",
        "Shape of the parameter surface around the optimum.",
        "Isolated optima often fit noise.",
        "Small parameter changes may destroy performance.",
        "See the heat-map.",
        "Prefer parameters on a plateau.",
        "Narrow genuine effects exist.",
    ),
    (
        "QP-SENS-002",
        "Unstable parameter ranking",
        "sensitivity",
        W,
        "WARN when the IS/OOS rank correlation across the grid is negative.",
        "Spearman correlation of first-half vs second-half Sharpe across the grid.",
        "If rankings reverse, selection adds no information.",
        "Selected parameters may underperform.",
        "See the IS/OOS comparison.",
        "Use validation-based selection.",
        "Two halves only.",
    ),
]
for rid, name, cat, sev, policy, desc, why, impact, inv, fix, lim in _VALUES:
    _add(
        id=rid,
        name=name,
        category=cat,
        analysis="statistical"
        if cat in {"statistics", "validation", "regime", "sensitivity"}
        else "runtime",
        max_severity=sev,
        severity_policy=policy,
        description=desc,
        rationale=why,
        impact=impact,
        investigate=inv,
        remediation=fix,
        limitations=lim,
    )

for rid, name, cat in (
    ("QP-STATIC", "Static analysis not run", "static"),
    ("QP-RUNTIME", "Runtime checks not run", "causality"),
):
    _add(
        id=rid,
        name=name,
        category=cat,
        analysis="meta",
        max_severity=INFO_,
        severity_policy="INFO; records that a check could not run.",
        description="Records checks that were skipped for lack of inputs.",
        rationale="",
        impact="",
        investigate="",
        remediation="",
        limitations="",
    )

REGISTRY: dict[str, RuleSpec] = {r.id: r for r in _RULES}
if len(REGISTRY) != len(_RULES):  # pragma: no cover - guards accidental duplicates
    raise RuntimeError("Duplicate rule ids in the registry")


def list_rules(category: str | None = None) -> list[RuleSpec]:
    """All rules, optionally filtered by category, in registry order."""
    return [r for r in _RULES if category is None or r.category == category]


def get_rule(rule_id: str) -> RuleSpec:
    """Metadata for one rule id (case-insensitive)."""
    key = rule_id.strip().upper()
    if key in REGISTRY:
        return REGISTRY[key]
    close = difflib.get_close_matches(key, list(REGISTRY), n=3)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise QuantProofInputError(
        f"Unknown rule id {rule_id!r}.{hint} Run `quantproof rules` for the list."
    )


def has_rule(rule_id: str) -> bool:
    """True if ``rule_id`` is registered (exact, case-sensitive)."""
    return rule_id in REGISTRY


_CATEGORY_TITLES = {
    "data": ("Data validation", "../methodology/data-validation.md"),
    "static": ("Static analysis", "../methodology/lookahead-detection.md"),
    "causality": ("Runtime causality", "../methodology/runtime-causality.md"),
    "leakage": ("Leakage", "../methodology/lookahead-detection.md"),
    "execution": ("Execution", "../methodology/transaction-costs.md"),
    "statistics": ("Statistics", "../methodology/sharpe-and-psr.md"),
    "validation": ("Validation", "../methodology/temporal-cross-validation.md"),
    "regime": ("Regimes", "../methodology/regimes.md"),
    "sensitivity": ("Parameter sensitivity", "../methodology/sensitivity.md"),
}


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def rules_markdown() -> str:
    """The rule catalogue as Markdown (``docs/api/rules.md`` is generated from this)."""
    out = [
        "# Rule catalogue",
        "",
        "<!-- Generated by `quantproof rules --markdown > docs/api/rules.md`;"
        " do not edit by hand. A test fails when this file is out of date. -->",
        "",
        "Every rule id QuantProof can emit, with its maximum severity and the exact severity "
        "policy. The same data is available as `quantproof rules --json` and "
        "`quantproof.rules.list_rules()`. *Analysis* is the level at which the rule works: "
        "`data`, `static` (source code), `runtime` (executes the strategy), `statistical` or "
        "`meta`. Confidence attached to findings is *analysis confidence* — how sure the "
        "analyzer is that the matched pattern means what the rule says — not a statistical "
        "probability.",
        "",
    ]
    categories = list(dict.fromkeys(r.category for r in _RULES))
    for cat in categories:
        title, link = _CATEGORY_TITLES.get(cat, (cat.title(), ""))
        out += [f"## {title}" + (f" — [methodology]({link})" if link else ""), ""]
        out += [
            "| ID | Name | Analysis | Max severity | Severity policy |",
            "|---|---|---|---|---|",
        ]
        for r in list_rules(cat):
            out.append(
                f"| `{r.id}` | {_md(r.name)} | {r.analysis} | {r.max_severity.value} | "
                f"{_md(r.severity_policy)} |"
            )
        out.append("")
        for r in list_rules(cat):
            if r.analysis == "meta":
                continue
            out += [f"### `{r.id}` {r.name}", ""]
            for label, value in (
                ("What it checks", r.description),
                ("Why it matters", r.rationale),
                ("Potential impact", r.impact),
                ("How to investigate", r.investigate),
                ("Remediation", r.remediation),
                ("Limitations", r.limitations),
            ):
                if value:
                    out.append(f"- **{label}:** {value}")
            if r.example:
                out += ["", "```python", r.example.rstrip(), "```"]
            out.append("")
    return "\n".join(out).rstrip() + "\n"
