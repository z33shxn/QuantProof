"""Static rules QP001–QP015.

Every rule is a small class with metadata (id, title, why it matters, remediation)
and a ``check`` method that receives a pre-built
:class:`~quantproof.audit.static.analyzer.ModuleContext`. New rules are added
with the :func:`register` decorator; no other module needs to change.

Severity policy
---------------
* ``FAIL`` is used only when a pattern establishes future information reaching
  a strategy output or model features, or another unambiguous validity violation.
* ``WARN`` is used when the pattern is usually a problem but context can make it
  valid (each message says which context).
* ``INFO`` records context (e.g. a detected search size) without judgement.
"""

from __future__ import annotations

import ast
import itertools
import math
import re
from abc import ABC, abstractmethod
from typing import ClassVar

from quantproof.audit.models import Category, Finding, Location
from quantproof.audit.severity import Confidence, Severity
from quantproof.audit.static.analyzer import (
    COST_TOKEN_RE,
    FIT_METHODS,
    FUTURE_NAME_RE,
    LABEL_NAME_RE,
    NOT_LITERAL,
    RETURN_NAME_RE,
    SIGNAL_NAME_RE,
    TRAIN_NAME_RE,
    CallSite,
    ModuleContext,
    TaintOrigin,
    TaintSink,
    dotted_name,
    get_kwarg,
    is_negative_expr,
    literal,
    names_in,
    string_keys_in,
)


class StaticRule(ABC):
    """Base class for static rules."""

    id: ClassVar[str]
    title: ClassVar[str]
    pass_title: ClassVar[str]
    why: ClassVar[str]
    remediation: ClassVar[str]

    @abstractmethod
    def check(self, ctx: ModuleContext) -> list[Finding]:
        """Return findings for this rule (empty if nothing was detected)."""

    def finding(
        self,
        ctx: ModuleContext,
        node: ast.AST | None,
        severity: Severity,
        message: str,
        *,
        confidence: Confidence = Confidence.MEDIUM,
        evidence: dict[str, object] | None = None,
        title: str | None = None,
    ) -> Finding:
        return Finding(
            id=self.id,
            category=Category.STATIC,
            severity=severity,
            title=title or self.title,
            message=message,
            evidence=evidence or {},
            location=ctx.location(node) if node is not None else Location(file=ctx.filename),
            why_it_matters=self.why,
            recommendation=self.remediation,
            confidence=confidence,
        )

    def passed(self, filename: str) -> Finding:
        return Finding(
            id=self.id,
            category=Category.STATIC,
            severity=Severity.PASS,
            title=self.pass_title,
            message=f"{self.pass_title} in {filename}.",
            location=Location(file=filename),
        )


RULES: list[StaticRule] = []


def register(cls: type[StaticRule]) -> type[StaticRule]:
    """Class decorator adding a rule instance to :data:`RULES`."""
    if any(r.id == cls.id for r in RULES):
        raise ValueError(f"Duplicate static rule id {cls.id}")
    RULES.append(cls())
    return cls


def _sinks_for(ctx: ModuleContext, origin: TaintOrigin, kinds: set[str]) -> list[str]:
    return [s.detail for s in ctx.sinks if s.kind in kinds and any(o is origin for o in s.origins)]


def _assigned_label(ctx: ModuleContext, node: ast.AST) -> str | None:
    for anc in ctx.ancestors(node):
        if isinstance(anc, (ast.Assign, ast.AnnAssign)):
            targets = anc.targets if isinstance(anc, ast.Assign) else [anc.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    return t.id
                if isinstance(t, ast.Subscript):
                    key = literal(t.slice)
                    if isinstance(key, str):
                        return key
            return None
        if isinstance(anc, ast.stmt):
            return None
    return None


# --------------------------------------------------------------------- QP001
@register
class NegativeShift(StaticRule):
    id = "QP001"
    title = "Look-ahead: negative temporal shift"
    pass_title = "No negative temporal shift detected"
    why = (
        "shift(-k), diff(-k) and pct_change(-k) move observations from the future into the "
        "current row. If the result feeds a signal or a feature, the backtest uses information "
        "that was not available at decision time."
    )
    remediation = (
        "Use negative shifts only to build prediction labels that never feed features or "
        "signals. For signals, use shift(k) with k ≥ 0 so each row uses only past data."
    )

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out: list[Finding] = []
        for origin in ctx.origins:
            if origin.rule != "QP001":
                continue
            node = origin.node
            assert isinstance(node, ast.Call)
            period = node.args[0] if node.args else get_kwarg(node, "periods")
            _neg, certain = is_negative_expr(period)
            signal_sinks = _sinks_for(ctx, origin, {"signal"})
            feature_sinks = _sinks_for(ctx, origin, {"feature"})
            label = _assigned_label(ctx, node)
            label_only = (
                not signal_sinks
                and not feature_sinks
                and (
                    _sinks_for(ctx, origin, {"label"}) != []
                    or (label is not None and LABEL_NAME_RE.match(label) is not None)
                )
            )
            evidence = {
                "period_literal": certain,
                "reaches": signal_sinks + feature_sinks,
                "assigned_to": label,
            }
            if signal_sinks:
                out.append(
                    self.finding(
                        ctx,
                        node,
                        Severity.FAIL,
                        f"{origin.description} reaches strategy output ({signal_sinks[0]}).",
                        confidence=Confidence.HIGH if certain else Confidence.MEDIUM,
                        evidence=evidence,
                    )
                )
            elif feature_sinks:
                out.append(
                    self.finding(
                        ctx,
                        node,
                        Severity.FAIL,
                        f"{origin.description} is used as a model input ({feature_sinks[0]}).",
                        confidence=Confidence.HIGH if certain else Confidence.MEDIUM,
                        evidence=evidence,
                    )
                )
            elif label_only:
                out.append(
                    self.finding(
                        ctx,
                        node,
                        Severity.INFO,
                        f"{origin.description} builds a prediction label"
                        + (f" ('{label}')" if label else "")
                        + ". This is legitimate only if the label never feeds features or signals.",
                        confidence=Confidence.MEDIUM,
                        evidence=evidence,
                        title="Negative shift used for label construction",
                    )
                )
            else:
                out.append(
                    self.finding(
                        ctx,
                        node,
                        Severity.WARN,
                        f"{origin.description} reads future observations"
                        + (f" (assigned to '{label}')" if label else "")
                        + ". It was not traced to a signal or feature, but verify it is only "
                        "used as a label.",
                        confidence=Confidence.MEDIUM,
                        evidence=evidence,
                    )
                )
        return out


# --------------------------------------------------------------------- QP002
@register
class CenteredRolling(StaticRule):
    id = "QP002"
    title = "Look-ahead: centered rolling window"
    pass_title = "No centered rolling window detected"
    why = (
        "rolling(..., center=True) labels each window by its middle observation, so the value "
        "at time t averages roughly window/2 future observations."
    )
    remediation = "Use trailing windows (center=False, the default)."

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        for c in ctx.calls_named("rolling"):
            center = get_kwarg(c.node, "center")
            value = literal(center)
            if value is True:
                out.append(
                    self.finding(
                        ctx,
                        c.node,
                        Severity.FAIL,
                        "rolling() is called with center=True; each value uses future observations.",
                        confidence=Confidence.HIGH,
                    )
                )
            elif center is not None and value is NOT_LITERAL:
                out.append(
                    self.finding(
                        ctx,
                        c.node,
                        Severity.WARN,
                        "rolling() is called with a non-literal `center` argument; confirm it is "
                        "False for any series used in signals.",
                        confidence=Confidence.LOW,
                    )
                )
        return out


# --------------------------------------------------------------------- QP003
@register
class FutureOrientedConstruction(StaticRule):
    id = "QP003"
    title = "Suspicious future-oriented construction"
    pass_title = "No future-oriented construction detected"
    why = (
        "Look-ahead index arithmetic (x[i + 1] inside a time loop), np.roll (which wraps the end "
        "of the series to the start) and future-named variables used as inputs commonly leak "
        "information from later bars."
    )
    remediation = (
        "Index only bars ≤ i when computing the decision for bar i. Replace np.roll with "
        "shift() (which inserts NaN instead of wrapping). Rename or remove future-named inputs."
    )

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out: list[Finding] = []
        for origin in ctx.origins:
            if origin.rule != "QP003":
                continue
            reaches = _sinks_for(ctx, origin, {"signal", "feature"})
            sev = Severity.FAIL if reaches else Severity.WARN
            msg = origin.description
            msg += (
                f" reaches strategy output/features ({reaches[0]})."
                if reaches
                else " reads data beyond the current bar; confirm it is used only for fills or labels."
            )
            out.append(
                self.finding(
                    ctx,
                    origin.node,
                    sev,
                    msg,
                    confidence=Confidence.HIGH if reaches else Confidence.LOW,
                    evidence={"reaches": reaches},
                )
            )
        # np.roll with positive shift wraps the last observations to the front.
        for c in ctx.calls:
            if c.name == "roll" and c.qualname.startswith(("numpy", "np")):
                shift = c.node.args[1] if len(c.node.args) > 1 else get_kwarg(c.node, "shift")
                neg, _ = is_negative_expr(shift)
                if not neg:
                    out.append(
                        self.finding(
                            ctx,
                            c.node,
                            Severity.WARN,
                            "np.roll wraps the final observations to the start of the array, "
                            "placing future values in the first rows.",
                            confidence=Confidence.MEDIUM,
                        )
                    )
        # Future-named identifiers used as model inputs.
        for c in ctx.calls:
            if c.name in FIT_METHODS | {"predict", "predict_proba"} and c.node.args:
                arg = c.node.args[0]
                ids = names_in(arg) | string_keys_in(arg)
                for name in ("X",):
                    ids.discard(name)
                const_ids: set[str] = set()
                for n in names_in(arg):
                    const = ctx.module_constants.get(n)
                    if isinstance(const, (list, tuple)):
                        const_ids.update(x for x in const if isinstance(x, str))
                future = sorted(i for i in ids | const_ids if FUTURE_NAME_RE.search(i))
                if future and not any(
                    o.node is s.node for s in ctx.sinks for o in s.origins if s.node is c.node
                ):
                    out.append(
                        self.finding(
                            ctx,
                            c.node,
                            Severity.WARN,
                            f"Model input in .{c.name}() references future-named field(s): "
                            f"{', '.join(future)}.",
                            confidence=Confidence.LOW,
                            evidence={"fields": future},
                        )
                    )
        return out


# --------------------------------------------------------------------- QP004
@register
class RandomTrainTestSplit(StaticRule):
    id = "QP004"
    title = "Random train/test split on (possibly) temporal data"
    pass_title = "No shuffled train/test split detected"
    why = (
        "train_test_split shuffles by default. With time-series data, shuffled splits put future "
        "observations in the training set and overlapping labels on both sides of the split, "
        "inflating out-of-sample performance. Context-dependent: it is fine for i.i.d. data."
    )
    remediation = (
        "Use train_test_split(..., shuffle=False), a walk-forward split, or "
        "quantproof.validation.PurgedKFold / CPCV."
    )

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        for c in ctx.calls_named("train_test_split"):
            shuffle = literal(get_kwarg(c.node, "shuffle"))
            if shuffle is False:
                continue
            explicit = shuffle is True
            out.append(
                self.finding(
                    ctx,
                    c.node,
                    Severity.WARN,
                    "train_test_split "
                    + ("with shuffle=True" if explicit else "with default shuffle=True")
                    + ". If rows are ordered in time, the test set is not out-of-sample.",
                    confidence=Confidence.MEDIUM,
                )
            )
        return out


# --------------------------------------------------------------------- QP005
_RANDOM_CV = {
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "GroupShuffleSplit",
    "StratifiedKFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "LeaveOneOut",
    "LeavePOut",
}
_SEARCH = {
    "cross_val_score",
    "cross_validate",
    "cross_val_predict",
    "GridSearchCV",
    "RandomizedSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
}


@register
class RandomizedCV(StaticRule):
    id = "QP005"
    title = "Randomized / non-temporal cross-validation"
    pass_title = "No randomized cross-validation detected"
    why = (
        "K-fold variants that shuffle (or that train on folds located after the test fold) "
        "let models learn from the future. With overlapping labels, test information also leaks "
        "into training. Context-dependent: harmless for genuinely i.i.d. samples."
    )
    remediation = (
        "Use TimeSeriesSplit for forward-only evaluation, or quantproof.validation.PurgedKFold "
        "/ CPCV with an embargo when labels overlap."
    )

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        for c in ctx.calls:
            if c.name == "KFold":
                shuffle = literal(get_kwarg(c.node, "shuffle"))
                if shuffle is True:
                    out.append(
                        self.finding(
                            ctx,
                            c.node,
                            Severity.WARN,
                            "KFold(shuffle=True) randomizes fold membership across time.",
                            confidence=Confidence.MEDIUM,
                        )
                    )
                else:
                    out.append(
                        self.finding(
                            ctx,
                            c.node,
                            Severity.INFO,
                            "Unshuffled KFold keeps contiguous folds but still trains on data "
                            "after each test fold; without purging/embargo, overlapping labels leak.",
                            confidence=Confidence.LOW,
                        )
                    )
            elif c.name in _RANDOM_CV:
                out.append(
                    self.finding(
                        ctx,
                        c.node,
                        Severity.WARN,
                        f"{c.name} does not respect temporal order.",
                        confidence=Confidence.MEDIUM,
                    )
                )
            elif c.name in _SEARCH:
                cv = get_kwarg(c.node, "cv")
                cv_val = literal(cv)
                if cv is None or isinstance(cv_val, int):
                    out.append(
                        self.finding(
                            ctx,
                            c.node,
                            Severity.INFO,
                            f"{c.name} uses the default K-fold splitter"
                            + (f" (cv={cv_val})" if isinstance(cv_val, int) else "")
                            + ", which ignores temporal order.",
                            confidence=Confidence.LOW,
                        )
                    )
        return out


# --------------------------------------------------------------------- QP006
_TRANSFORMERS = {
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "PCA",
    "KernelPCA",
    "TruncatedSVD",
    "SelectKBest",
    "SelectPercentile",
    "VarianceThreshold",
    "RFE",
    "RFECV",
    "SelectFromModel",
    "KBinsDiscretizer",
}


def _transformer_fits(ctx: ModuleContext) -> list[tuple[CallSite, ast.expr]]:
    """(fit call, fitted data expression) for preprocessing transformers."""
    bound: dict[str, str] = {}
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = node.value.func
            cname = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            if cname in _TRANSFORMERS:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        bound[t.id] = cname
    out = []
    for c in ctx.calls:
        if c.name not in {"fit", "fit_transform"} or not c.node.args:
            continue
        recv = c.receiver
        is_transformer = False
        if isinstance(recv, ast.Name) and recv.id in bound:
            is_transformer = True
        elif isinstance(recv, ast.Call):
            f = recv.func
            cname = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            is_transformer = cname in _TRANSFORMERS
        if is_transformer:
            out.append((c, c.node.args[0]))
    return out


@register
class FitBeforeSplit(StaticRule):
    id = "QP006"
    title = "Leakage: preprocessing fitted before the train/test split"
    pass_title = "No fit-before-split leakage detected"
    why = (
        "Fitting a scaler, imputer, PCA or feature selector on the full dataset lets statistics "
        "of the test period (means, variances, selected features) shape the training data."
    )
    remediation = (
        "Split first, fit transformers on the training fold only, then transform the test fold "
        "(an sklearn Pipeline inside a temporal CV loop does this automatically)."
    )

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        for call, data in _transformer_fits(ctx):
            data_ids = names_in(data) | string_keys_in(data)
            if any(TRAIN_NAME_RE.search(i) for i in data_ids):
                continue
            later_splits = [
                s
                for s in ctx.splits
                if getattr(s.node, "lineno", 0) > call.node.lineno
                and (s.scope == call.scope or s.scope == "<module>" or call.scope == "<module>")
            ]
            earlier_splits = [
                s
                for s in ctx.splits
                if getattr(s.node, "lineno", 0) < call.node.lineno
                and (s.scope == call.scope or s.scope == "<module>")
            ]
            if later_splits and not earlier_splits:
                out.append(
                    self.finding(
                        ctx,
                        call.node,
                        Severity.FAIL,
                        f".{call.name}() on '{ast.unparse(data)}' happens before the first "
                        f"train/test split (line {getattr(later_splits[0].node, 'lineno', '?')}).",
                        confidence=Confidence.MEDIUM,
                        evidence={"split_line": getattr(later_splits[0].node, "lineno", None)},
                    )
                )
        return out


# --------------------------------------------------------------------- QP007
_LOCAL_STAT_TOKENS = {"rolling", "expanding", "ewm", "groupby", "resample", "cummax", "cummin"}


def _is_global_stat(node: ast.AST, methods: set[str]) -> ast.Call | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in methods
    ):
        recv_ids = names_in(node.func.value)
        if recv_ids & _LOCAL_STAT_TOKENS:
            return None
        if any(TRAIN_NAME_RE.search(i) for i in recv_ids | string_keys_in(node.func.value)):
            return None
        return node
    return None


@register
class FullSampleNormalization(StaticRule):
    id = "QP007"
    title = "Full-sample normalization"
    pass_title = "No full-sample normalization detected"
    why = (
        "(x - x.mean()) / x.std(), min-max scaling over the whole series, or scalers fitted on all "
        "data use statistics computed over the entire sample, including the future. A z-score at "
        "time t then depends on prices after t."
    )
    remediation = (
        "Use trailing statistics (x.rolling(n).mean(), x.expanding().std()) or fit the scaler on "
        "the training window only. Cross-sectional normalization within a date is fine."
    )

    def _cross_sectional(self, ctx: ModuleContext, node: ast.AST) -> bool:
        for anc in ctx.ancestors(node):
            if isinstance(anc, ast.Call):
                ids = names_in(anc.func)
                if "groupby" in ids or (
                    isinstance(anc.func, ast.Attribute)
                    and anc.func.attr == "apply"
                    and literal(get_kwarg(anc, "axis")) in (1, "columns")
                ):
                    return True
        return False

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out: list[Finding] = []
        seen: set[int] = set()
        for node in ast.walk(ctx.tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                continue
            num = node.left
            if not (isinstance(num, ast.BinOp) and isinstance(num.op, ast.Sub)):
                continue
            center = _is_global_stat(num.right, {"mean", "median", "min"})
            if center is None:
                continue
            if self._cross_sectional(ctx, node):
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            out.append(
                self.finding(
                    ctx,
                    node,
                    Severity.WARN,
                    f"'{ast.unparse(node)[:80]}' normalizes with statistics of the full sample.",
                    confidence=Confidence.MEDIUM,
                )
            )
        for c in ctx.calls:
            if c.name in {"zscore", "scale", "minmax_scale", "robust_scale"} and (
                c.qualname.startswith(("scipy", "sklearn")) or c.name == "zscore"
            ):
                if self._cross_sectional(ctx, c.node):
                    continue
                out.append(
                    self.finding(
                        ctx,
                        c.node,
                        Severity.WARN,
                        f"{c.name}() standardizes using full-sample statistics.",
                        confidence=Confidence.MEDIUM,
                    )
                )
        split_lines = [getattr(s.node, "lineno", 0) for s in ctx.splits]
        for call, data in _transformer_fits(ctx):
            ids = names_in(data) | string_keys_in(data)
            if any(TRAIN_NAME_RE.search(i) for i in ids):
                continue
            if split_lines:
                continue  # handled by QP006 (fit before split) or fitted after a split
            out.append(
                self.finding(
                    ctx,
                    call.node,
                    Severity.WARN,
                    f"Transformer .{call.name}() on '{ast.unparse(data)}' with no train/test "
                    "split anywhere in the file: statistics come from the full sample.",
                    confidence=Confidence.MEDIUM,
                )
            )
        return out


# --------------------------------------------------------------------- QP008
@register
class TargetLeakage(StaticRule):
    id = "QP008"
    title = "Target leakage: target included in features"
    pass_title = "No direct target leakage detected"
    why = (
        "If the prediction target (or a column it is copied from) is part of the feature matrix, "
        "the model can simply read the answer; in-sample and cross-validated scores become "
        "meaningless."
    )
    remediation = "Exclude the target column (and anything derived from it) from the features."

    def _str_list(self, ctx: ModuleContext, node: ast.AST) -> list[str] | None:
        value = literal(node)
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            return list(value)
        if isinstance(node, ast.Name):
            const = ctx.module_constants.get(node.id)
            if isinstance(const, (list, tuple)) and all(isinstance(v, str) for v in const):
                return list(const)
        return None

    def check(self, ctx: ModuleContext) -> list[Finding]:
        target_keys: dict[str, str] = {}  # target variable name -> column key
        features: dict[str, tuple[ast.AST, list[str] | None, list[str] | None]] = {}
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            value = node.value
            if isinstance(value, ast.Subscript):
                key = literal(value.slice)
                if key is NOT_LITERAL and isinstance(value.slice, ast.Name):
                    key = ctx.module_constants.get(value.slice.id, NOT_LITERAL)
                if isinstance(key, str) and LABEL_NAME_RE.match(tgt.id):
                    target_keys[tgt.id] = key
                cols = self._str_list(ctx, value.slice)
                if cols is not None:
                    features[tgt.id] = (node, cols, None)
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "drop"
            ):
                dropped = None
                cols_kw = get_kwarg(value, "columns")
                if cols_kw is not None:
                    dropped = self._str_list(ctx, cols_kw)
                    single = literal(cols_kw)
                    if isinstance(single, str):
                        dropped = [single]
                elif value.args:
                    dropped = self._str_list(ctx, value.args[0])
                    single = literal(value.args[0])
                    if isinstance(single, str):
                        dropped = [single]
                if dropped is not None:
                    features[tgt.id] = (node, None, dropped)
        out = []
        for c in ctx.calls:
            if c.name not in FIT_METHODS or len(c.node.args) < 2:
                continue
            x_node, y_node = c.node.args[0], c.node.args[1]
            if ast.dump(x_node) == ast.dump(y_node):
                out.append(
                    self.finding(
                        ctx,
                        c.node,
                        Severity.FAIL,
                        f".{c.name}() receives the same expression as features and target.",
                        confidence=Confidence.HIGH,
                    )
                )
                continue
            y_key = None
            if isinstance(y_node, ast.Name):
                y_key = target_keys.get(y_node.id)
            elif isinstance(y_node, ast.Subscript):
                k = literal(y_node.slice)
                y_key = k if isinstance(k, str) else None
            if y_key is None:
                continue
            x_cols: list[str] | None = None
            x_dropped: list[str] | None = None
            if isinstance(x_node, ast.Name) and x_node.id in features:
                _, x_cols, x_dropped = features[x_node.id]
            elif isinstance(x_node, ast.Subscript):
                x_cols = self._str_list(ctx, x_node.slice)
            if x_cols is not None and y_key in x_cols:
                out.append(
                    self.finding(
                        ctx,
                        c.node,
                        Severity.FAIL,
                        f"Target column '{y_key}' is included in the feature matrix passed to "
                        f".{c.name}().",
                        confidence=Confidence.HIGH,
                        evidence={"target": y_key, "features": x_cols},
                    )
                )
            elif x_dropped is not None and y_key not in x_dropped:
                out.append(
                    self.finding(
                        ctx,
                        c.node,
                        Severity.FAIL,
                        f"Features are built with .drop({x_dropped}) but the target column "
                        f"'{y_key}' is not dropped.",
                        confidence=Confidence.MEDIUM,
                        evidence={"target": y_key, "dropped": x_dropped},
                    )
                )
        return out


# ---------------------------------------------------------- QP009 / QP010 / QP011
def _has_positive_shift(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "shift"
        ):
            period = sub.args[0] if sub.args else get_kwarg(sub, "periods")
            if period is None:
                return True  # shift() defaults to 1
            value = literal(period)
            if isinstance(value, (int, float)) and value > 0:
                return True
            if value is NOT_LITERAL and not is_negative_expr(period)[0]:
                return True
    return False


def _operand_role(node: ast.AST) -> str | None:
    ids = names_in(node) | string_keys_in(node)
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"pct_change", "diff"}
        ):
            return "returns"
    if any(SIGNAL_NAME_RE.search(i) for i in ids):
        return "signal"
    if any(RETURN_NAME_RE.search(i) for i in ids):
        return "returns"
    return None


def _strategy_return_products(ctx: ModuleContext) -> list[tuple[ast.expr, ast.expr, ast.expr]]:
    """Expressions multiplying a signal-like operand by a returns-like operand."""
    out: list[tuple[ast.expr, ast.expr, ast.expr]] = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            a, b = node.left, node.right
            ra, rb = _operand_role(a), _operand_role(b)
            if ra == "signal" and rb == "returns":
                out.append((node, a, b))
            elif rb == "signal" and ra == "returns":
                out.append((node, b, a))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"mul", "multiply"}
            and node.args
        ):
            a, b = node.func.value, node.args[0]
            ra, rb = _operand_role(a), _operand_role(b)
            if {ra, rb} == {"signal", "returns"}:
                sig, ret = (a, b) if ra == "signal" else (b, a)
                out.append((node, sig, ret))
    return out


def _declared_execution(ctx: ModuleContext) -> dict[str, object] | None:
    value = ctx.module_constants.get("EXECUTION")
    return value if isinstance(value, dict) else None


@register
class SameBarExecution(StaticRule):
    id = "QP009"
    title = "Same-bar signal/execution assumption"
    pass_title = "No undocumented same-bar execution detected"
    why = (
        "A signal computed from a bar's close and filled at that same close assumes zero latency "
        "and an exact closing fill. Short-horizon strategies often lose most of their edge once a "
        "realistic delay is introduced."
    )
    remediation = (
        "Declare the assumption explicitly (EXECUTION = {'signal_lag': ...}) and test with at "
        "least one bar of execution lag or next-open fills."
    )

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        declared = _declared_execution(ctx)
        if declared is not None:
            lag = declared.get("signal_lag")
            if isinstance(lag, (int, float)) and lag == 0:
                node: ast.AST | None = next(
                    (
                        s
                        for s in ctx.tree.body
                        if isinstance(s, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "EXECUTION" for t in s.targets)
                    ),
                    None,
                )
                out.append(
                    self.finding(
                        ctx,
                        node,
                        Severity.WARN,
                        "EXECUTION declares signal_lag=0: trades are filled at the close that "
                        "generated the signal.",
                        confidence=Confidence.HIGH,
                        evidence={"declared": declared},
                    )
                )
            return out
        documented = any(
            re.search(r"assum|same[- ]bar|market[- ]on[- ]close|\bmoc\b", c, re.I)
            for c in ctx.comments.values()
        )
        for node, sig, _ret in _strategy_return_products(ctx):
            if _has_positive_shift(sig):
                shift_one = any(
                    isinstance(s, ast.Call)
                    and isinstance(s.func, ast.Attribute)
                    and s.func.attr == "shift"
                    and literal(s.args[0] if s.args else get_kwarg(s, "periods")) in (1, None)
                    for s in ast.walk(sig)
                )
                if shift_one:
                    out.append(
                        self.finding(
                            ctx,
                            node,
                            Severity.INFO if documented else Severity.WARN,
                            "Positions lagged by exactly one bar are multiplied by close-to-close "
                            "returns: the trade is assumed to fill at the same close that produced "
                            "the signal"
                            + (" (assumption documented in comments)." if documented else "."),
                            confidence=Confidence.LOW,
                        )
                    )
        return out


@register
class MissingExecutionLag(StaticRule):
    id = "QP010"
    title = "Missing execution lag (signal × same-period return)"
    pass_title = "Signals are lagged before being multiplied by returns"
    why = (
        "Multiplying a signal at time t by the return realized over the same period t "
        "(close[t-1] → close[t]) credits the strategy with a move that already happened when "
        "the signal was computed from close[t]. This is one of the most common look-ahead bugs."
    )
    remediation = "Lag positions before applying returns: returns * signal.shift(1) (or more)."

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        for node, sig, ret in _strategy_return_products(ctx):
            if _has_positive_shift(sig):
                continue
            # A negative shift on the returns side is the alternative alignment and is fine.
            if any(
                isinstance(s, ast.Call)
                and isinstance(s.func, ast.Attribute)
                and s.func.attr == "shift"
                and is_negative_expr(s.args[0] if s.args else get_kwarg(s, "periods"))[0]
                for s in ast.walk(ret)
            ):
                continue
            if _has_positive_shift(ret):
                continue  # both shifted (e.g. returns.shift(1) * signal) – ambiguous, skip
            out.append(
                self.finding(
                    ctx,
                    node,
                    Severity.FAIL,
                    f"'{ast.unparse(node)[:80]}' multiplies an un-lagged signal by same-period "
                    "returns.",
                    confidence=Confidence.MEDIUM,
                )
            )
        return out


@register
class MissingTransactionCosts(StaticRule):
    id = "QP011"
    title = "Transaction cost model missing"
    pass_title = "Transaction costs referenced or not applicable"
    why = (
        "Gross backtest returns ignore commissions, spread, slippage and impact. High-turnover "
        "strategies can look profitable gross and lose money net."
    )
    remediation = (
        "Subtract costs proportional to turnover (quantproof.execution.TransactionCostModel) or "
        "declare cost assumptions in EXECUTION = {'commission_bps': ..., 'slippage_bps': ...}."
    )

    def check(self, ctx: ModuleContext) -> list[Finding]:
        products = _strategy_return_products(ctx)
        declared = _declared_execution(ctx)
        if not products and declared is None:
            return []
        if declared is not None:
            cost_keys = [k for k in declared if COST_TOKEN_RE.search(str(k))]
            nonzero = [k for k in cost_keys if declared.get(k) not in (0, 0.0, None)]
            if nonzero:
                return []
            node: ast.AST | None = next(
                (
                    s
                    for s in ctx.tree.body
                    if isinstance(s, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "EXECUTION" for t in s.targets)
                ),
                None,
            )
            return [
                self.finding(
                    ctx,
                    node,
                    Severity.WARN,
                    "EXECUTION declares no non-zero transaction cost assumptions"
                    + (f" ({', '.join(cost_keys)} = 0)." if cost_keys else "."),
                    confidence=Confidence.HIGH,
                    evidence={"declared": declared},
                )
            ]
        identifiers = ctx.all_identifiers()
        if any(COST_TOKEN_RE.search(i) for i in identifiers):
            return []
        node = products[0][0]
        return [
            self.finding(
                ctx,
                node,
                Severity.WARN,
                "Strategy returns are computed but no cost, commission, fee, slippage or spread "
                "term appears anywhere in the file.",
                confidence=Confidence.MEDIUM,
            )
        ]


# --------------------------------------------------------------------- QP012
@register
class FutureReturnsAsFeatures(StaticRule):
    id = "QP012"
    title = "Future returns used as model features"
    pass_title = "No future-derived model features detected"
    why = (
        "A feature built from future prices or returns (typically the label itself, or a shifted "
        "copy) gives the model perfect foresight during fitting and evaluation."
    )
    remediation = "Build features from information available at or before each row's timestamp."

    def check(self, ctx: ModuleContext) -> list[Finding]:
        # One finding per set of origins, anchored at the first sink; later sinks
        # reached by the same future-derived data are listed in the evidence.
        groups: dict[tuple[int, ...], list[TaintSink]] = {}
        for sink in ctx.sinks:
            if sink.kind != "feature":
                continue
            key = tuple(sorted(id(o.node) for o in sink.origins))
            groups.setdefault(key, []).append(sink)
        out = []
        for sinks in groups.values():
            sinks.sort(key=lambda s: getattr(s.node, "lineno", 0))
            first = sinks[0]
            lines = sorted({getattr(o.node, "lineno", 0) for o in first.origins})
            sink_lines = [getattr(s.node, "lineno", 0) for s in sinks]
            out.append(
                self.finding(
                    ctx,
                    first.node,
                    Severity.FAIL,
                    f"Future-derived data (from line(s) {', '.join(map(str, lines))}) is "
                    f"{first.detail}"
                    + (
                        f" and reaches {len(sinks) - 1} further model call(s) (lines "
                        f"{', '.join(map(str, sink_lines[1:]))})."
                        if len(sinks) > 1
                        else "."
                    ),
                    confidence=Confidence.HIGH,
                    evidence={"origin_lines": lines, "sink_lines": sink_lines},
                )
            )
        return out


# --------------------------------------------------------------------- QP013
def _grid_size(value: object) -> int | None:
    if isinstance(value, dict):
        sizes = [len(v) if isinstance(v, (list, tuple, range)) else 1 for v in value.values()]
        return math.prod(sizes) if sizes else 1
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, dict) for v in value):
        total = 0
        for v in value:
            size = _grid_size(v)
            total += size or 0
        return total
    return None


def _iter_len(node: ast.AST, ctx: ModuleContext) -> int | None:
    value = literal(node)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(node, ast.Name):
        const = ctx.module_constants.get(node.id)
        if isinstance(const, (list, tuple, set)):
            return len(const)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        args = [literal(a) for a in node.args]
        if all(isinstance(a, int) for a in args) and args:
            return len(range(*args))
    if (
        isinstance(node, ast.Call)
        and dotted_name(node.func) in {"np.arange", "numpy.arange", "np.linspace", "numpy.linspace"}
        and node.args
    ):
        args = [literal(a) for a in node.args]
        if all(isinstance(a, (int, float)) for a in args):
            if dotted_name(node.func) in {"np.linspace", "numpy.linspace"}:
                return int(args[2]) if len(args) >= 3 else 50
            return len(range(*[int(a) for a in args]))  # approximate for float steps
    return None


@register
class ExcessiveSearch(StaticRule):
    id = "QP013"
    title = "Excessive hyper-parameter search"
    pass_title = "No large hyper-parameter search detected"
    why = (
        "The best of many backtests is biased upward even when no variant has skill. The number "
        "of trials must be reported and corrected for (e.g. with the Deflated Sharpe Ratio)."
    )
    remediation = (
        "Record every trial, pass the count to QuantProof (statistics.trials) so the Deflated "
        "Sharpe Ratio accounts for it, and prefer coarse, economically motivated grids."
    )

    def check(self, ctx: ModuleContext) -> list[Finding]:
        threshold = ctx.config.max_trials_threshold
        found: list[tuple[ast.AST | None, int, str]] = []
        for c in ctx.calls:
            if c.name in {"GridSearchCV", "HalvingGridSearchCV"}:
                grid = c.node.args[1] if len(c.node.args) > 1 else get_kwarg(c.node, "param_grid")
                value = literal(grid) if grid is not None else NOT_LITERAL
                if value is NOT_LITERAL and isinstance(grid, ast.Name):
                    value = ctx.module_constants.get(grid.id, NOT_LITERAL)
                size = _grid_size(value)
                if size:
                    found.append((c.node, size, f"{c.name} grid"))
            elif c.name in {"RandomizedSearchCV", "HalvingRandomSearchCV"}:
                n_iter = literal(get_kwarg(c.node, "n_iter"))
                found.append((c.node, n_iter if isinstance(n_iter, int) else 10, f"{c.name}"))
            elif c.name == "optimize" and get_kwarg(c.node, "n_trials") is not None:
                n = literal(get_kwarg(c.node, "n_trials"))
                if isinstance(n, int):
                    found.append((c.node, n, "optimizer n_trials"))
            elif c.name == "product" and c.qualname.startswith("itertools"):
                lens = [_iter_len(a, ctx) for a in c.node.args]
                if lens and all(n is not None for n in lens):
                    found.append((c.node, math.prod(n for n in lens if n), "itertools.product"))
        for name in ("PARAM_GRID", "PARAMETER_GRID", "param_grid"):
            if name in ctx.module_constants:
                size = _grid_size(ctx.module_constants[name])
                if size:
                    node: ast.AST | None = next(
                        (
                            s
                            for s in ctx.tree.body
                            if isinstance(s, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == name for t in s.targets)
                        ),
                        None,
                    )
                    found.append((node, size, name))
        # Directly nested for-loops over constant iterables.
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.For) and not any(
                isinstance(a, ast.For) for a in itertools.islice(ctx.ancestors(node), 50)
            ):
                total, depth = 1, 0
                cur: ast.For | None = node
                while isinstance(cur, ast.For):
                    n = _iter_len(cur.iter, ctx)
                    if n is None:
                        break
                    total *= n
                    depth += 1
                    inner = [s for s in cur.body if isinstance(s, ast.For)]
                    cur = inner[0] if len(inner) == 1 else None
                if depth >= 2:
                    found.append((node, total, f"{depth} nested parameter loops"))
        out = []
        for node, size, what in found:
            sev = Severity.WARN if size > threshold else Severity.INFO
            out.append(
                self.finding(
                    ctx,
                    node,
                    sev,
                    f"{what} evaluates about {size} configurations"
                    + (
                        f" (threshold {threshold}). Report this count as statistics.trials."
                        if sev is Severity.WARN
                        else "; pass this count as statistics.trials for the Deflated Sharpe Ratio."
                    ),
                    confidence=Confidence.MEDIUM,
                    evidence={"trials": size, "source": what, "threshold": threshold},
                    title=self.title if sev is Severity.WARN else "Parameter search detected",
                )
            )
        return out


# --------------------------------------------------------------------- QP014
@register
class NoOutOfSample(StaticRule):
    id = "QP014"
    title = "No explicit out-of-sample evaluation"
    pass_title = "Out-of-sample evaluation present or not applicable"
    why = (
        "Fitting or optimizing on the full sample and reporting the result measures how well the "
        "procedure fits history, not how well it generalizes."
    )
    remediation = (
        "Hold out a final test period, or evaluate with walk-forward / purged cross-validation."
    )

    def check(self, ctx: ModuleContext) -> list[Finding]:
        fits = [c for c in ctx.calls if c.name in {"fit", "fit_transform"} and c.node.args]
        model_fits = [c for c in fits if not any(c is t for t, _ in _transformer_fits(ctx))]
        # A declared PARAM_GRID is QuantProof's contract: the audit itself evaluates the grid
        # out of sample (walk-forward, CPCV, PBO), so it does not count as an unvalidated search.
        has_search = any(c.name in _SEARCH | {"optimize", "minimize", "product"} for c in ctx.calls)
        if not model_fits and not has_search:
            return []
        if ctx.splits:
            return []
        oos_tokens = re.compile(r"test|oos|out_of_sample|holdout|validation|walk", re.I)
        if any(oos_tokens.search(i) for i in ctx.all_identifiers()):
            return []
        node = model_fits[0].node if model_fits else None
        return [
            self.finding(
                ctx,
                node,
                Severity.WARN,
                "Models are fitted or parameters searched, but no train/test split, walk-forward "
                "or cross-validation construct was found.",
                confidence=Confidence.MEDIUM,
            )
        ]


# --------------------------------------------------------------------- QP015
@register
class NonCausalTransform(StaticRule):
    id = "QP015"
    title = "Potentially non-causal transformation"
    pass_title = "No non-causal transformation detected"
    why = (
        "Zero-phase filters (filtfilt), Savitzky-Golay and Gaussian smoothing, HP filters, "
        "seasonal decomposition, FFT filtering, linear interpolation and backward fills all use "
        "observations after t to produce the value at t."
    )
    remediation = (
        "Use one-sided (causal) filters such as lfilter or ewm, re-estimate decompositions on "
        "expanding windows, and forward-fill only."
    )

    _ALWAYS: ClassVar[dict[str, str]] = {
        "filtfilt": "zero-phase filtfilt",
        "sosfiltfilt": "zero-phase sosfiltfilt",
        "savgol_filter": "Savitzky-Golay filter (centered)",
        "gaussian_filter1d": "Gaussian filter (centered)",
        "hpfilter": "Hodrick-Prescott filter (two-sided)",
        "seasonal_decompose": "seasonal decomposition (two-sided)",
        "STL": "STL decomposition (two-sided)",
        "detrend": "full-sample detrending",
        "bfill": "backward fill",
        "backfill": "backward fill",
    }

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        for c in ctx.calls:
            desc = self._ALWAYS.get(c.name)
            if c.name == "detrend" and not c.qualname.startswith("scipy"):
                desc = None
            if c.name == "fillna":
                method = literal(get_kwarg(c.node, "method"))
                if method in {"bfill", "backfill"}:
                    desc = "fillna(method='bfill')"
            if c.name == "interpolate":
                method = literal(get_kwarg(c.node, "method"))
                direction = literal(get_kwarg(c.node, "limit_direction"))
                if method not in {"pad", "ffill"} or direction in {"backward", "both"}:
                    desc = (
                        f"interpolate(method={method if method is not NOT_LITERAL else 'linear'!r})"
                    )
            if c.name == "convolve" and c.qualname.startswith(("numpy", "np")):
                mode = literal(get_kwarg(c.node, "mode"))
                if len(c.node.args) > 2:
                    mode = literal(c.node.args[2])
                if mode == "same":
                    desc = "np.convolve(mode='same') (centered kernel)"
            if c.qualname.startswith(("numpy.fft", "np.fft", "scipy.fft")):
                desc = "FFT-based transform over the whole sample"
            if desc:
                out.append(
                    self.finding(
                        ctx,
                        c.node,
                        Severity.WARN,
                        f"{desc} uses future observations to compute past values.",
                        confidence=Confidence.MEDIUM,
                    )
                )
        return out


def list_rules() -> list[dict[str, str]]:
    """Metadata for every registered static rule (for docs and the CLI)."""
    return [
        {"id": r.id, "title": r.title, "why": r.why, "remediation": r.remediation} for r in RULES
    ]
