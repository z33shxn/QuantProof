"""Static rules QP001–QP015.

Every rule is a small class with metadata (id, title, why it matters, remediation)
and a ``check`` method that receives a pre-built
:class:`~quantproof.analyzers.static.analyzer.ModuleContext`. New rules are added
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
from collections.abc import Mapping
from typing import ClassVar

from quantproof.analyzers.static.analyzer import (
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
    TaintSink,
    dotted_name,
    get_kwarg,
    is_cross_sectional,
    is_full_sample_normalization,
    is_negative_expr,
    literal,
    names_in,
    non_causal_description,
    string_keys_in,
)
from quantproof.results import Category, Finding, Location, Usage
from quantproof.rules import RuleSpec, get_rule
from quantproof.severity import Confidence, Severity


class StaticRule(ABC):
    """Base class for static rules. Metadata comes from :mod:`quantproof.rules`."""

    id: ClassVar[str]

    @property
    def spec(self) -> RuleSpec:
        return get_rule(self.id)

    @property
    def title(self) -> str:
        return self.spec.name

    @property
    def pass_title(self) -> str:
        return self.spec.pass_title or f"{self.spec.name}: nothing detected"

    @property
    def why(self) -> str:
        return self.spec.rationale

    @property
    def remediation(self) -> str:
        return self.spec.remediation

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
        evidence: Mapping[str, object] | None = None,
        title: str | None = None,
        usage: Usage | None = None,
    ) -> Finding:
        return Finding(
            id=self.id,
            category=Category.STATIC,
            severity=severity,
            title=title or self.title,
            message=message,
            evidence=dict(evidence or {}),
            location=ctx.location(node) if node is not None else Location(file=ctx.filename),
            why_it_matters=self.why,
            recommendation=self.remediation,
            confidence=confidence,
            usage=usage,
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


def classify_use(ctx: ModuleContext, node: ast.AST) -> tuple[Usage | None, list[str]]:
    """How the future-dated value produced at ``node`` is used.

    Returns ``(LIVE_DECISION, details)`` if it reaches a signal or model input,
    ``(LABEL_OR_ANALYSIS, details)`` if it only reaches labels/targets, and
    ``(None, [])`` if its use could not be determined.
    """
    live = [s.detail for s in ctx.sinks_reached(node, {"signal", "feature"})]
    if live:
        return Usage.LIVE_DECISION, live
    labels = [s.detail for s in ctx.sinks_reached(node, {"label"})]
    name = _assigned_label(ctx, node)
    if labels or (name is not None and LABEL_NAME_RE.match(name)):
        return Usage.LABEL_OR_ANALYSIS, labels or [f"assigned to '{name}'"]
    return None, []


def timing_finding(
    rule: StaticRule,
    ctx: ModuleContext,
    node: ast.AST,
    what: str,
    *,
    unknown_severity: Severity = Severity.WARN,
    live_severity: Severity = Severity.FAIL,
    live_confidence: Confidence = Confidence.HIGH,
) -> Finding:
    """Finding for a future-dated construct, with severity decided by how it is used."""
    usage, details = classify_use(ctx, node)
    evidence = {"construct": what, "reaches": details, "usage": usage.value if usage else None}
    if usage is Usage.LIVE_DECISION:
        return rule.finding(
            ctx,
            node,
            live_severity,
            f"{what} reaches a live decision ({details[0]}).",
            confidence=live_confidence,
            evidence=evidence,
            usage=usage,
        )
    if usage is Usage.LABEL_OR_ANALYSIS:
        return rule.finding(
            ctx,
            node,
            Severity.INFO,
            f"{what} is used for label/analysis construction ({details[0]}); legitimate as long "
            "as it never feeds features or signals.",
            confidence=Confidence.MEDIUM,
            evidence=evidence,
            usage=usage,
            title=f"{rule.title} (label/analysis use)",
        )
    return rule.finding(
        ctx,
        node,
        unknown_severity,
        f"{what} uses future observations; it was not traced to a signal, model input or label. "
        "Verify it is only used for analysis or labels.",
        confidence=Confidence.MEDIUM,
        evidence=evidence,
    )


# --------------------------------------------------------------------- QP001
@register
class NegativeShift(StaticRule):
    id = "QP001"

    def check(self, ctx: ModuleContext) -> list[Finding]:
        return [
            timing_finding(self, ctx, o.node, o.description)
            for o in ctx.origins
            if o.rule == "QP001"
        ]


# --------------------------------------------------------------------- QP002
@register
class CenteredRolling(StaticRule):
    id = "QP002"

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = [
            timing_finding(self, ctx, o.node, o.description)
            for o in ctx.origins
            if o.rule == "QP002"
        ]
        for c in ctx.calls_named("rolling"):
            center = get_kwarg(c.node, "center")
            if center is not None and literal(center) is NOT_LITERAL:
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

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out: list[Finding] = []
        for origin in ctx.origins:
            if origin.rule == "QP003":
                out.append(
                    timing_finding(
                        self,
                        ctx,
                        origin.node,
                        origin.description,
                        live_confidence=Confidence.MEDIUM,
                    )
                )
        # np.roll with a positive shift wraps the last observations to the front.
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
        # Future-named identifiers used as model inputs (name-based, low confidence).
        for c in ctx.calls:
            if c.name in FIT_METHODS | {"predict", "predict_proba"} and c.node.args:
                arg = c.node.args[0]
                ids = (names_in(arg) | string_keys_in(arg)) - {"X"}
                for n in names_in(arg):
                    const = ctx.module_constants.get(n)
                    if isinstance(const, (list, tuple)):
                        ids.update(x for x in const if isinstance(x, str))
                future = sorted(i for i in ids if FUTURE_NAME_RE.search(i))
                traced = any(s.node is c.node for s in ctx.sinks if s.kind == "feature")
                if future and not traced:
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


def _fit_outputs(ctx: ModuleContext, call: CallSite) -> set[str]:
    """Names holding the fitted data or its transform (for linking a fit to a later split)."""
    out = set(names_in(call.node.args[0]))
    label = _assigned_label(ctx, call.node)
    if label:
        out.add(label)
    recv = call.receiver
    if isinstance(recv, ast.Name):
        for c in ctx.calls:
            if (
                c.name == "transform"
                and isinstance(c.receiver, ast.Name)
                and c.receiver.id == recv.id
            ):
                lbl = _assigned_label(ctx, c.node)
                if lbl:
                    out.add(lbl)
    return out


@register
class FitBeforeSplit(StaticRule):
    id = "QP006"

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        for call, data in _transformer_fits(ctx):
            data_ids = names_in(data) | string_keys_in(data)
            if any(TRAIN_NAME_RE.search(i) for i in data_ids):
                continue
            later = [
                s
                for s in ctx.splits
                if getattr(s.node, "lineno", 0) > call.node.lineno
                and (s.scope == call.scope or s.scope == "<module>" or call.scope == "<module>")
            ]
            earlier = [
                s
                for s in ctx.splits
                if getattr(s.node, "lineno", 0) < call.node.lineno
                and (s.scope == call.scope or s.scope == "<module>")
            ]
            if not later or earlier:
                continue
            fitted = _fit_outputs(ctx, call)
            linked = [s for s in later if names_in(s.node) & fitted]
            first = (linked or later)[0]
            line = getattr(first.node, "lineno", None)
            if linked:
                out.append(
                    self.finding(
                        ctx,
                        call.node,
                        Severity.FAIL,
                        f".{call.name}() on '{ast.unparse(data)}' is fitted before the data is split "
                        f"(line {line}); test-period statistics shape the training data.",
                        confidence=Confidence.MEDIUM,
                        evidence={
                            "split_line": line,
                            "linked_names": sorted(fitted & names_in(first.node)),
                        },
                    )
                )
            else:
                out.append(
                    self.finding(
                        ctx,
                        call.node,
                        Severity.WARN,
                        f".{call.name}() on '{ast.unparse(data)}' happens before a train/test split "
                        f"(line {line}); the fitted data could not be linked to the split, so confirm "
                        "it is not the data being split.",
                        confidence=Confidence.LOW,
                        evidence={"split_line": line},
                    )
                )
        return out


# --------------------------------------------------------------------- QP007
@register
class FullSampleNormalization(StaticRule):
    id = "QP007"

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out: list[Finding] = []
        nodes = [n for n in ast.walk(ctx.tree) if is_full_sample_normalization(n)]
        nodes += [
            c.node
            for c in ctx.calls
            if c.name in {"zscore", "scale", "minmax_scale", "robust_scale"}
            and (c.qualname.startswith(("scipy", "sklearn")) or c.name == "zscore")
        ]
        for node in nodes:
            if is_cross_sectional(ctx, node):
                continue
            usage, details = classify_use(ctx, node)
            text = ast.unparse(node)[:80]
            msg = f"'{text}' normalizes with statistics of the full sample"
            if usage is Usage.LIVE_DECISION:
                msg += (
                    f" and reaches a live decision ({details[0]}). If the series is time-indexed this "
                    "is look-ahead; the runtime causality test gives definitive evidence."
                )
            else:
                msg += "."
            out.append(
                self.finding(
                    ctx,
                    node,
                    Severity.WARN,
                    msg,
                    confidence=Confidence.MEDIUM,
                    evidence={"reaches": details},
                    usage=usage,
                )
            )
        split_lines = [getattr(s.node, "lineno", 0) for s in ctx.splits]
        for call, data in _transformer_fits(ctx):
            ids = names_in(data) | string_keys_in(data)
            if any(TRAIN_NAME_RE.search(i) for i in ids) or split_lines:
                continue
            out.append(
                self.finding(
                    ctx,
                    call.node,
                    Severity.WARN,
                    f"Transformer .{call.name}() on '{ast.unparse(data)}' with no train/test split "
                    "anywhere in the file: statistics come from the full sample.",
                    confidence=Confidence.MEDIUM,
                )
            )
        return out


# --------------------------------------------------------------------- QP008
@register
class TargetLeakage(StaticRule):
    id = "QP008"

    def _str_list(self, ctx: ModuleContext, node: ast.AST) -> list[str] | None:
        value = literal(node)
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            return list(value)
        if isinstance(node, ast.Name):
            const = ctx.module_constants.get(node.id)
            if isinstance(const, (list, tuple)) and all(isinstance(v, str) for v in const):
                return list(const)
        return None

    @staticmethod
    def _fit_target_keys(ctx: ModuleContext) -> set[str]:
        out: set[str] = set()
        for c in ctx.calls:
            if (
                c.name in FIT_METHODS
                and len(c.node.args) > 1
                and isinstance(c.node.args[1], ast.Subscript)
            ):
                key = literal(c.node.args[1].slice)
                if isinstance(key, str):
                    out.add(key)
        return out

    @staticmethod
    def _target_copies(ctx: ModuleContext, targets: set[str]) -> dict[str, str]:
        """Columns assigned as direct copies of a target column (no temporal transform)."""
        temporal = {"shift", "rolling", "expanding", "ewm", "diff", "pct_change", "resample"}
        copies: dict[str, str] = {}
        changed = True
        while changed:
            changed = False
            for node in ast.walk(ctx.tree):
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                    continue
                tgt = node.targets[0]
                key = literal(tgt.slice) if isinstance(tgt, ast.Subscript) else NOT_LITERAL
                if not isinstance(key, str) or key in targets or key in copies:
                    continue
                if names_in(node.value) & temporal:
                    continue
                for ref in string_keys_in(node.value):
                    origin = ref if ref in targets else copies.get(ref)
                    if origin:
                        copies[key] = origin
                        changed = True
                        break
        return copies

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
        copies = self._target_copies(ctx, set(target_keys.values()) | self._fit_target_keys(ctx))
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
            copied = sorted(k for k in (x_cols or []) if copies.get(k) == y_key)
            if x_cols is not None and copied and y_key not in x_cols:
                out.append(
                    self.finding(
                        ctx,
                        c.node,
                        Severity.FAIL,
                        f"Feature column(s) {copied} are copies of the target '{y_key}' passed to "
                        f".{c.name}().",
                        confidence=Confidence.MEDIUM,
                        evidence={"target": y_key, "copies": copied},
                    )
                )
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


CLOSE_COLUMNS = {"close", "adj_close", "price", "Close", "Adj Close"}
OPEN_COLUMNS = {"open", "Open"}


def _definitions(ctx: ModuleContext) -> dict[str, ast.expr]:
    """Last assignment to each name / string column key (flow-insensitive)."""
    out: dict[str, ast.expr] = {}
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                out[tgt.id] = node.value
            elif isinstance(tgt, ast.Subscript):
                key = literal(tgt.slice)
                if isinstance(key, str):
                    out[key] = node.value
    return out


def price_provenance(
    expr: ast.AST, defs: dict[str, ast.expr], depth: int = 0, seen: frozenset[str] = frozenset()
) -> set[str]:
    """Price columns an expression depends on *at the same bar*.

    References inside a positive ``shift(k)`` are prior bars and are ignored; names and
    column keys are resolved through their definitions (up to 6 levels).
    """
    found: set[str] = set()

    def visit(node: ast.AST) -> None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "shift"
        ):
            period = node.args[0] if node.args else get_kwarg(node, "periods")
            value = literal(period) if period is not None else 1
            if isinstance(value, (int, float)) and value > 0:
                return  # prior bar only
        if isinstance(node, ast.Subscript):
            key = literal(node.slice)
            if isinstance(key, str):
                if key in CLOSE_COLUMNS | OPEN_COLUMNS:
                    found.add("open" if key in OPEN_COLUMNS else "close")
                elif key in defs and key not in seen and depth < 6:
                    found.update(price_provenance(defs[key], defs, depth + 1, seen | {key}))
        elif isinstance(node, ast.Attribute) and node.attr in CLOSE_COLUMNS | OPEN_COLUMNS:
            found.add("open" if node.attr in OPEN_COLUMNS else "close")
        elif isinstance(node, ast.Name) and node.id in defs and node.id not in seen and depth < 6:
            found.update(price_provenance(defs[node.id], defs, depth + 1, seen | {node.id}))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(expr)
    return found


@register
class MissingExecutionLag(StaticRule):
    id = "QP010"

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        defs = _definitions(ctx)
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
            sig_src = price_provenance(sig, defs)
            ret_src = price_provenance(ret, defs)
            evidence = {"signal_uses": sorted(sig_src), "returns_use": sorted(ret_src)}
            text = ast.unparse(node)[:80]
            if sig_src == {"open"} and "open" in ret_src:
                continue  # decided at the open, earns open → close: same-bar is legitimate
            if "close" in sig_src and ret_src == {"close"}:
                out.append(
                    self.finding(
                        ctx,
                        node,
                        Severity.FAIL,
                        f"'{text}' multiplies a signal computed from the bar's close by the return "
                        "that ends at that same close.",
                        confidence=Confidence.HIGH,
                        evidence=evidence,
                        usage=Usage.LIVE_DECISION,
                    )
                )
            else:
                out.append(
                    self.finding(
                        ctx,
                        node,
                        Severity.WARN,
                        f"'{text}' multiplies an un-lagged signal by same-period returns. Execution "
                        "semantics could not be determined automatically (the price inputs of the "
                        "signal or the returns could not be traced).",
                        confidence=Confidence.LOW,
                        evidence=evidence,
                    )
                )
        return out


@register
class MissingTransactionCosts(StaticRule):
    id = "QP011"

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

    def check(self, ctx: ModuleContext) -> list[Finding]:
        # One finding per set of origins, anchored at the first sink; later sinks
        # reached by the same future-derived data are listed in the evidence.
        # Only definite future constructs count; heuristic full-sample normalization (QP007)
        # is reported by its own rule.
        definite = {"QP001", "QP002", "QP003", "QP015"}
        groups: dict[tuple[int, ...], list[TaintSink]] = {}
        for sink in ctx.sinks:
            origins = [o for o in sink.origins if o.rule in definite]
            if sink.kind != "feature" or not origins:
                continue
            key = tuple(sorted(id(o.node) for o in origins))
            groups.setdefault(key, []).append(TaintSink(sink.kind, sink.node, origins, sink.detail))
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
                    usage=Usage.LIVE_DECISION,
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

    def check(self, ctx: ModuleContext) -> list[Finding]:
        out = []
        for c in ctx.calls:
            desc = non_causal_description(c.node, c.name, c.qualname)
            if desc:
                out.append(
                    timing_finding(
                        self,
                        ctx,
                        c.node,
                        f"{desc} uses future observations to compute past values;",
                        live_confidence=Confidence.MEDIUM,
                    )
                )
        return out


def list_rules() -> list[RuleSpec]:
    """Registry metadata for every registered static rule."""
    return [r.spec for r in RULES]
