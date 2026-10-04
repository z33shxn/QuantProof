"""AST-based static analysis of Python research code.

The analyzer parses source code once into a :class:`ModuleContext` that exposes
pre-computed facts (resolved call names, scopes, a future-information taint
analysis, split events, ...). Each rule in :mod:`quantproof.analyzers.static.rules`
inspects that context and emits findings.

Static analysis is heuristic by nature: it reasons about syntax, not about the
values flowing at runtime. Findings therefore carry a confidence level, and the
runtime future-perturbation test (:mod:`quantproof.analyzers.causal`) provides
complementary behavioural evidence.

Inline suppression: append ``# quantproof: ignore[QP001]`` (or ``ignore`` for all
rules) to a line to suppress findings on it. Suppressions are reported as INFO.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantproof.config import StaticConfig
from quantproof.results import Category, Finding, Location
from quantproof.severity import Severity

SIGNAL_NAME_RE = re.compile(
    r"(^|_)(signal|signals|sig|position|positions|pos|weight|weights|holding|holdings|"
    r"exposure|target_position|alloc|allocation)(_|$|\d)",
    re.IGNORECASE,
)
RETURN_NAME_RE = re.compile(r"(^|_)(ret|rets|return|returns|pct|pnl_base|r)(_|$|\d)", re.I)
LABEL_NAME_RE = re.compile(
    r"^(y|y_\w+|\w+_y|target\w*|label\w*|\w*_target|\w*_label|outcome\w*)$", re.IGNORECASE
)
FUTURE_NAME_RE = re.compile(
    r"(^|_)(future|fwd|forward|next|lead|leading|tomorrow|ahead|tplus\d*)(_|$|\d)", re.IGNORECASE
)
TRAIN_NAME_RE = re.compile(r"train|insample|in_sample|\bis_", re.IGNORECASE)
COST_TOKEN_RE = re.compile(
    r"cost|commission|fee|slippage|spread|bps|tcost|transaction|brokerage|impact", re.IGNORECASE
)
ENTRY_FUNCTIONS = {
    "generate_signals",
    "compute_signals",
    "signals",
    "strategy",
    "generate_positions",
    "compute_positions",
    "positions",
    "run_strategy",
    "backtest",
    "run_backtest",
}
SHIFT_LIKE = {"shift", "diff", "pct_change"}
FIT_METHODS = {"fit", "fit_transform", "partial_fit"}
PREDICT_METHODS = {"predict", "predict_proba", "decision_function", "transform", "score"}


def dotted_name(node: ast.AST) -> str | None:
    """``a.b.c`` for Name/Attribute chains, else None."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def literal(node: ast.AST | None) -> Any:
    """Safely evaluate a literal expression (numbers, strings, lists, negatives).

    Returns the sentinel :data:`NOT_LITERAL` if the node is not a literal.
    """
    if node is None:
        return NOT_LITERAL
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return NOT_LITERAL


class _NotLiteral:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NOT_LITERAL"


NOT_LITERAL: Any = _NotLiteral()


def get_kwarg(call: ast.Call, name: str) -> ast.expr | None:
    """Keyword argument expression by name."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def is_negative_expr(
    node: ast.AST | None, constants: dict[str, Any] | None = None
) -> tuple[bool, bool]:
    """Return ``(is_negative, certain)`` for a shift-period expression.

    ``shift(-1)`` → (True, True); ``shift(-n)`` → (True, False); ``shift(1)`` → (False, True).
    A bare name bound to a module-level numeric constant (``HORIZON = -5``) is resolved
    through ``constants``.
    """
    if node is None:
        return False, True
    value = literal(node)
    if value is NOT_LITERAL and isinstance(node, ast.Name) and constants:
        value = constants.get(node.id, NOT_LITERAL)
    if value is not NOT_LITERAL:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0, True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return True, False
    return False, False


def names_in(node: ast.AST) -> set[str]:
    """All identifiers referenced (Name ids and attribute names) inside ``node``."""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
    return out


def string_keys_in(node: ast.AST) -> set[str]:
    """String constants used as subscript keys or attribute names inside ``node``."""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript):
            key = literal(sub.slice)
            if isinstance(key, str):
                out.add(key)
            elif isinstance(key, (list, tuple)):
                out.update(k for k in key if isinstance(k, str))
    return out


@dataclass
class CallSite:
    """A call expression with its resolved name."""

    node: ast.Call
    name: str  # last component, e.g. "shift"
    qualname: str  # best-effort dotted name with import aliases resolved
    receiver: ast.expr | None  # object a method is called on
    scope: str


@dataclass
class TaintOrigin:
    """Where a future-information value was introduced."""

    rule: str  # QP001 (negative shift) or QP003 (other future-oriented construction)
    node: ast.AST
    description: str


@dataclass
class TaintSink:
    """A future-derived value reaching a sensitive place."""

    kind: str  # "signal", "feature", "label"
    node: ast.AST
    origins: list[TaintOrigin]
    detail: str


@dataclass
class SplitEvent:
    """A train/test split or out-of-sample construct."""

    node: ast.AST
    kind: str
    scope: str


@dataclass
class ModuleContext:
    """Everything the rules need, computed once per file."""

    source: str
    filename: str
    tree: ast.Module
    lines: list[str]
    imports: dict[str, str] = field(default_factory=dict)
    calls: list[CallSite] = field(default_factory=list)
    parents: dict[int, ast.AST] = field(default_factory=dict)
    scope_of: dict[int, str] = field(default_factory=dict)
    sinks: list[TaintSink] = field(default_factory=list)
    origins: list[TaintOrigin] = field(default_factory=list)
    splits: list[SplitEvent] = field(default_factory=list)
    comments: dict[int, str] = field(default_factory=dict)
    module_constants: dict[str, Any] = field(default_factory=dict)
    aliases: dict[str, tuple[str, bool]] = field(default_factory=dict)
    summaries: dict[str, FunctionSummary] = field(default_factory=dict)
    config: StaticConfig = field(default_factory=StaticConfig)
    # Extra function names whose return value is the strategy's live signal (e.g. the name
    # of a callable passed to ``audit``), in addition to ENTRY_FUNCTIONS.
    entry_points: frozenset[str] = frozenset()

    def origin_for(self, node: ast.AST) -> TaintOrigin | None:
        """The taint origin registered for ``node`` (if the node is a source)."""
        for o in self.origins:
            if o.node is node:
                return o
        return None

    def sinks_reached(self, node: ast.AST, kinds: set[str]) -> list[TaintSink]:
        """Sinks of the given kinds reached by the origin at ``node``."""
        origin = self.origin_for(node)
        if origin is None:
            return []
        return [s for s in self.sinks if s.kind in kinds and any(o is origin for o in s.origins)]

    # ------------------------------------------------------------------ helpers
    def location(self, node: ast.AST) -> Location:
        line = getattr(node, "lineno", None)
        snippet = self.lines[line - 1].strip() if line and line <= len(self.lines) else None
        return Location(
            file=self.filename,
            line=line,
            column=(getattr(node, "col_offset", None) or 0) + 1 if line else None,
            end_line=getattr(node, "end_lineno", None),
            snippet=snippet,
        )

    def calls_named(self, *names: str) -> Iterator[CallSite]:
        wanted = set(names)
        for c in self.calls:
            if c.name in wanted:
                yield c

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.parents.get(id(node))

    def ancestors(self, node: ast.AST) -> Iterator[ast.AST]:
        cur = self.parent(node)
        while cur is not None:
            yield cur
            cur = self.parent(cur)

    def all_identifiers(self) -> set[str]:
        out = names_in(self.tree)
        for sub in ast.walk(self.tree):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                out.add(sub.value)
            elif isinstance(sub, ast.keyword) and sub.arg:
                out.add(sub.arg)
            elif isinstance(sub, ast.FunctionDef):
                out.update(a.arg for a in sub.args.args + sub.args.kwonlyargs)
        return out

    def suppressed(self, line: int | None, rule_id: str) -> bool:
        if line is None:
            return False
        comment = self.comments.get(line, "")
        m = re.search(r"quantproof:\s*ignore(?:\[([A-Za-z0-9_,\- ]+)\])?", comment)
        if not m:
            return False
        if m.group(1) is None:
            return True
        return rule_id in {p.strip() for p in m.group(1).split(",")}


# ---------------------------------------------------------------------- building
def _collect_comments(source: str) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Comments only drive inline suppression; keep whatever was collected so far.
        return out
    return out


def _collect_imports(tree: ast.Module) -> dict[str, str]:
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return imports


def _build_parents_and_scopes(tree: ast.Module) -> tuple[dict[int, ast.AST], dict[int, str]]:
    parents: dict[int, ast.AST] = {}
    scopes: dict[int, str] = {}

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
            child_scope = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_scope = f"{scope}.{child.name}" if scope != "<module>" else child.name
            scopes[id(child)] = child_scope
            visit(child, child_scope)

    scopes[id(tree)] = "<module>"
    visit(tree, "<module>")
    return parents, scopes


def _resolve_call(node: ast.Call, imports: dict[str, str]) -> tuple[str, str, ast.expr | None]:
    func = node.func
    if isinstance(func, ast.Name):
        qual = imports.get(func.id, func.id)
        # Aliased imports (``from x import train_test_split as tts``) resolve to the real name.
        return qual.rsplit(".", 1)[-1], qual, None
    if isinstance(func, ast.Attribute):
        dotted = dotted_name(func)
        qual = func.attr
        if dotted:
            head, _, rest = dotted.partition(".")
            qual = f"{imports.get(head, head)}.{rest}" if rest else imports.get(head, head)
        return func.attr, qual, func.value
    return "<dynamic>", "<dynamic>", None


def _module_constants(tree: ast.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name):
                value = literal(stmt.value)
                if value is not NOT_LITERAL:
                    out[tgt.id] = value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            value = literal(stmt.value)
            if value is not NOT_LITERAL:
                out[stmt.target.id] = value
    return out


def build_context(
    source: str,
    filename: str = "<string>",
    config: StaticConfig | None = None,
    *,
    entry_points: frozenset[str] | set[str] | None = None,
) -> ModuleContext:
    """Parse source and pre-compute facts. Raises :class:`SyntaxError` on invalid code."""
    tree = ast.parse(source, filename=filename)
    ctx = ModuleContext(
        source=source,
        filename=filename,
        tree=tree,
        lines=source.splitlines(),
        config=config or StaticConfig(),
        entry_points=frozenset(entry_points or ()),
    )
    ctx.imports = _collect_imports(tree)
    ctx.parents, ctx.scope_of = _build_parents_and_scopes(tree)
    ctx.comments = _collect_comments(source)
    ctx.module_constants = _module_constants(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name, qual, recv = _resolve_call(node, ctx.imports)
            ctx.calls.append(CallSite(node, name, qual, recv, ctx.scope_of.get(id(node), "")))
    ctx.calls.sort(key=lambda c: (c.node.lineno, c.node.col_offset))
    ctx.splits = _collect_splits(ctx)
    ctx.aliases = _collect_aliases(tree)
    ctx.summaries = compute_summaries(ctx)
    _TaintAnalysis(ctx).run()
    return ctx


# ---------------------------------------------------------------- split events
SPLIT_FUNCTIONS = {
    "train_test_split",
    "TimeSeriesSplit",
    "PurgedKFold",
    "CPCV",
    "CombinatorialPurgedCV",
    "WalkForward",
    "walk_forward",
    "walk_forward_splits",
    "temporal_train_test_split",
    "cross_val_score",
    "cross_validate",
    "cross_val_predict",
    "GridSearchCV",
    "RandomizedSearchCV",
}


def _collect_splits(ctx: ModuleContext) -> list[SplitEvent]:
    events: list[SplitEvent] = []
    for c in ctx.calls:
        if c.name in SPLIT_FUNCTIONS or c.name == "split":
            events.append(SplitEvent(c.node, c.name, c.scope))
    for node in ast.walk(ctx.tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            tnames: set[str] = set()
            for t in targets:
                tnames |= names_in(t) | string_keys_in(t)
            value = node.value
            if value is None:
                continue
            if any(re.search(r"train|test|oos|holdout|valid", n, re.I) for n in tnames) and any(
                isinstance(s, (ast.Slice, ast.Compare)) for s in ast.walk(value)
            ):
                events.append(SplitEvent(node, "slice", ctx.scope_of.get(id(node), "")))
    events.sort(key=lambda e: (getattr(e.node, "lineno", 0), getattr(e.node, "col_offset", 0)))
    return events


# ----------------------------------------------------------------- taint flow
NON_CAUSAL_CALLS = {
    "filtfilt": "zero-phase filtfilt",
    "sosfiltfilt": "zero-phase sosfiltfilt",
    "savgol_filter": "Savitzky-Golay filter (centered)",
    "gaussian_filter1d": "Gaussian filter (centered)",
    "hpfilter": "Hodrick-Prescott filter (two-sided)",
    "seasonal_decompose": "seasonal decomposition (two-sided)",
    "STL": "STL decomposition (two-sided)",
    "bfill": "backward fill",
    "backfill": "backward fill",
}
LOCAL_STAT_TOKENS = {"rolling", "expanding", "ewm", "groupby", "resample", "cummax", "cummin"}


def non_causal_description(call: ast.Call, name: str, qual: str) -> str | None:
    """Description if ``call`` is a two-sided / non-causal transformation, else None."""
    desc = NON_CAUSAL_CALLS.get(name)
    if name == "detrend" and qual.startswith("scipy"):
        desc = "full-sample detrending"
    if name == "fillna":
        method = literal(get_kwarg(call, "method"))
        if method in {"bfill", "backfill"}:
            desc = "fillna(method='bfill')"
    if name == "interpolate":
        method = literal(get_kwarg(call, "method"))
        direction = literal(get_kwarg(call, "limit_direction"))
        if method not in {"pad", "ffill"} or direction in {"backward", "both"}:
            shown = method if method is not NOT_LITERAL else "linear"
            desc = f"interpolate(method={shown!r})"
    if name == "convolve" and qual.startswith(("numpy", "np")):
        mode = literal(get_kwarg(call, "mode"))
        if len(call.args) > 2:
            mode = literal(call.args[2])
        if mode == "same":
            desc = "np.convolve(mode='same') (centered kernel)"
    if qual.startswith(("numpy.fft", "np.fft", "scipy.fft")):
        desc = "FFT-based transform over the whole sample"
    return desc


def global_stat_call(node: ast.AST, methods: set[str]) -> ast.Call | None:
    """``x.mean()``-style call over a whole series (not rolling/expanding/train-only)."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in methods
    ):
        recv_ids = names_in(node.func.value)
        if recv_ids & LOCAL_STAT_TOKENS:
            return None
        if any(TRAIN_NAME_RE.search(i) for i in recv_ids | string_keys_in(node.func.value)):
            return None
        return node
    return None


def is_full_sample_normalization(node: ast.AST) -> bool:
    """``(x - x.mean()) / x.std()`` (or median/min centring) over a whole series."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
        return False
    num = node.left
    return (
        isinstance(num, ast.BinOp)
        and isinstance(num.op, ast.Sub)
        and global_stat_call(num.right, {"mean", "median", "min"}) is not None
    )


PANDAS_GROUP_TOKENS = {"groupby", "resample", "rolling", "expanding", "ewm"}


def is_cross_sectional(ctx: ModuleContext, node: ast.AST) -> bool:
    """True if ``node`` sits inside a groupby(...) call chain or ``apply(axis=1)``."""
    for anc in ctx.ancestors(node):
        if isinstance(anc, ast.Call):
            if "groupby" in names_in(anc.func):
                return True
            if (
                isinstance(anc.func, ast.Attribute)
                and anc.func.attr == "apply"
                and literal(get_kwarg(anc, "axis")) in (1, "columns")
            ):
                return True
    return False


@dataclass
class FunctionSummary:
    """What a helper function does with future information (Level-3 analysis)."""

    name: str
    returns: list[TaintOrigin] = field(default_factory=list)
    mutated_keys: dict[str, list[TaintOrigin]] = field(default_factory=dict)


def _collect_aliases(tree: ast.Module) -> dict[str, tuple[str, bool]]:
    """``lead = pd.Series.shift`` → {"lead": ("shift", True)} (unbound: first arg is the object)."""
    out: dict[str, tuple[str, bool]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt, value = node.targets[0], node.value
            if (
                isinstance(tgt, ast.Name)
                and isinstance(value, ast.Attribute)
                and value.attr in SHIFT_LIKE | {"rolling"}
            ):
                dotted = dotted_name(value) or ""
                head = dotted.split(".")[0]
                unbound = head in {"pd", "pandas", "Series", "DataFrame"} or dotted.endswith(
                    ("Series." + value.attr, "DataFrame." + value.attr)
                )
                out[tgt.id] = (value.attr, unbound)
    return out


class _TaintAnalysis:
    """Order-sensitive taint tracking of future information (analysis Levels 1–3).

    Sources
        negative ``shift``/``diff``/``pct_change`` (QP001), centered ``rolling`` (QP002),
        ``np.roll`` and ``x[i + k]`` in loops (QP003), full-sample normalization (QP007),
        two-sided transformations such as ``bfill``/``filtfilt`` (QP015) — including calls
        through simple aliases (``lead = pd.Series.shift``).
    Propagation
        assignments to names, string column keys (``df["x"] = ...``), attributes and
        ``DataFrame.assign``; through expressions, lambdas and comprehensions; through calls
        to helper functions defined in the same file (their *return value* and the columns
        they set on DataFrames passed to them). Each function has its own scope; module-level
        taint is visible inside functions.
    Sinks
        ``signal``: returned from a strategy entry function or assigned to a
        signal/position/weight name; ``feature``: first argument of fit/predict/transform;
        ``label``: target argument of fit or a name like ``y``/``target``/``label``.
    Boundary
        no cross-module analysis, no containers (lists/dicts of series), no dynamic
        attributes (``getattr``), no aliasing of DataFrame objects.
    """

    def __init__(
        self,
        ctx: ModuleContext,
        *,
        record: bool = True,
        summaries: dict[str, FunctionSummary] | None = None,
    ) -> None:
        self.ctx = ctx
        self.record = record
        self.summaries = summaries if summaries is not None else ctx.summaries
        self.loop_vars: list[set[str]] = []

    def run(self) -> None:
        self._block(self.ctx.tree.body, {}, {}, func=None)

    # ------------------------------------------------------------- origins
    def _origins_in(
        self,
        node: ast.AST,
        names: dict[str, list[TaintOrigin]],
        keys: dict[str, list[TaintOrigin]],
    ) -> list[TaintOrigin]:
        found: list[TaintOrigin] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                origin = self._source_call(sub)
                if origin:
                    found.append(origin)
                summary = self._summary_for(sub)
                if summary is not None:
                    found.extend(summary.returns)
            elif (
                isinstance(sub, ast.BinOp)
                and is_full_sample_normalization(sub)
                and not is_cross_sectional(self.ctx, sub)
            ):
                found.append(self._origin("QP007", sub, "full-sample normalization"))
            elif isinstance(sub, ast.Subscript):
                origin = self._lookahead_index(sub)
                if origin:
                    found.append(origin)
                for k in self._subscript_keys(sub):
                    found.extend(keys.get(k, []))
            elif isinstance(sub, ast.Name) and sub.id in names:
                found.extend(names[sub.id])
            elif isinstance(sub, ast.Attribute) and sub.attr in keys:
                found.extend(keys[sub.attr])
        seen: set[int] = set()
        unique = []
        for o in found:
            if id(o.node) not in seen:
                seen.add(id(o.node))
                unique.append(o)
        return unique

    def _subscript_keys(self, sub: ast.Subscript) -> list[str]:
        key = literal(sub.slice)
        if isinstance(key, str):
            return [key]
        if isinstance(key, (list, tuple)):
            return [k for k in key if isinstance(k, str)]
        if isinstance(sub.slice, ast.Name) and sub.slice.id in self.ctx.module_constants:
            const = self.ctx.module_constants[sub.slice.id]
            items = const if isinstance(const, (list, tuple)) else [const]
            return [k for k in items if isinstance(k, str)]
        return []

    def _summary_for(self, call: ast.Call) -> FunctionSummary | None:
        func = call.func
        if isinstance(func, ast.Name):
            return self.summaries.get(func.id)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in {"self", "cls"}
        ):
            return self.summaries.get(func.attr)
        return None

    def _source_call(self, call: ast.Call) -> TaintOrigin | None:
        name, qual, _ = _resolve_call(call, self.ctx.imports)
        args = list(call.args)
        alias = self.ctx.aliases.get(name) if isinstance(call.func, ast.Name) else None
        if alias is not None:
            name, unbound = alias
            if unbound:
                args = args[1:]
        if name in SHIFT_LIKE:
            period = args[0] if args else get_kwarg(call, "periods")
            neg, _certain = is_negative_expr(period, self.ctx.module_constants)
            if neg:
                via = (
                    f" (via alias '{call.func.id}')"
                    if alias and isinstance(call.func, ast.Name)
                    else ""
                )
                return self._origin("QP001", call, f"{name}() with a negative period{via}")
        if name == "rolling" and literal(get_kwarg(call, "center")) is True:
            return self._origin("QP002", call, "centered rolling window")
        if name == "roll" and qual.startswith(("numpy", "np")):
            shift = call.args[1] if len(call.args) > 1 else get_kwarg(call, "shift")
            neg, _ = is_negative_expr(shift, self.ctx.module_constants)
            if neg:
                return self._origin("QP003", call, "np.roll with a negative shift")
        if name in {"zscore", "scale", "minmax_scale", "robust_scale"} and (
            qual.startswith(("scipy", "sklearn")) or name == "zscore"
        ):
            return self._origin("QP007", call, f"{name}() over the full sample")
        desc = non_causal_description(call, name, qual)
        if desc:
            return self._origin("QP015", call, desc)
        return None

    def _lookahead_index(self, sub: ast.Subscript) -> TaintOrigin | None:
        if not self.loop_vars:
            return None
        idx = sub.slice
        if isinstance(idx, ast.BinOp) and isinstance(idx.op, ast.Add):
            loop_names = set().union(*self.loop_vars)
            left, right = idx.left, idx.right
            for a, b in ((left, right), (right, left)):
                if isinstance(a, ast.Name) and a.id in loop_names:
                    k = literal(b)
                    if isinstance(k, int) and k > 0:
                        return self._origin("QP003", sub, f"look-ahead index [{a.id} + {k}]")
        return None

    def _origin(self, rule: str, node: ast.AST, desc: str) -> TaintOrigin:
        for o in self.ctx.origins:
            if o.node is node:
                return o
        origin = TaintOrigin(rule, node, desc)
        self.ctx.origins.append(origin)
        return origin

    def _sink(self, kind: str, node: ast.AST, origins: list[TaintOrigin], detail: str) -> None:
        if self.record:
            self.ctx.sinks.append(TaintSink(kind, node, origins, detail))

    # ---------------------------------------------------------- statements
    def _assign_targets(
        self,
        targets: Iterable[ast.expr],
        origins: list[TaintOrigin],
        names: dict[str, list[TaintOrigin]],
        keys: dict[str, list[TaintOrigin]],
        stmt: ast.AST,
    ) -> None:
        for tgt in targets:
            if isinstance(tgt, (ast.Tuple, ast.List)):
                self._assign_targets(tgt.elts, origins, names, keys, stmt)
                continue
            label: str | None = None
            if isinstance(tgt, ast.Name):
                label = tgt.id
                if origins:
                    names[tgt.id] = origins
                else:
                    names.pop(tgt.id, None)
            elif isinstance(tgt, ast.Subscript):
                key = literal(tgt.slice)
                if isinstance(key, str):
                    label = key
                    if origins:
                        keys[key] = origins
                    else:
                        keys.pop(key, None)
            elif isinstance(tgt, ast.Attribute):
                label = tgt.attr
                if origins:
                    keys[tgt.attr] = origins
            if origins and label and SIGNAL_NAME_RE.search(label):
                self._sink("signal", stmt, origins, f"assigned to signal/position '{label}'")
            elif origins and label and LABEL_NAME_RE.match(label):
                self._sink("label", stmt, origins, f"assigned to '{label}'")

    def _check_calls(
        self,
        node: ast.AST,
        names: dict[str, list[TaintOrigin]],
        keys: dict[str, list[TaintOrigin]],
    ) -> None:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            name, _, _ = _resolve_call(sub, self.ctx.imports)
            summary = self._summary_for(sub)
            if summary is not None:
                for k, origins in summary.mutated_keys.items():
                    keys[k] = origins
            if (
                name == "transform"
                and isinstance(sub.func, ast.Attribute)
                and (names_in(sub.func.value) & PANDAS_GROUP_TOKENS)
            ):
                continue  # pandas groupby/rolling transform, not a model input
            if name in FIT_METHODS | PREDICT_METHODS and sub.args:
                origins = self._origins_in(sub.args[0], names, keys)
                if origins:
                    self._sink("feature", sub, origins, f"used as features in .{name}()")
                if name in FIT_METHODS and len(sub.args) > 1:
                    y_origins = self._origins_in(sub.args[1], names, keys)
                    if y_origins:
                        self._sink("label", sub, y_origins, f"used as target in .{name}()")
            elif name == "assign":
                for kw in sub.keywords:
                    if kw.arg is None:
                        continue
                    origins = self._origins_in(kw.value, names, keys)
                    if origins:
                        keys[kw.arg] = origins
                        if SIGNAL_NAME_RE.search(kw.arg):
                            self._sink("signal", sub, origins, f"assigned to '{kw.arg}'")

    def _block(
        self,
        body: list[ast.stmt],
        names: dict[str, list[TaintOrigin]],
        keys: dict[str, list[TaintOrigin]],
        func: str | None,
    ) -> None:
        for stmt in body:
            self._stmt(stmt, names, keys, func)

    def _stmt(
        self,
        stmt: ast.stmt,
        names: dict[str, list[TaintOrigin]],
        keys: dict[str, list[TaintOrigin]],
        func: str | None,
    ) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Each function has its own scope; module-level taint is visible inside it.
            self._block(stmt.body, dict(names), dict(keys), func=stmt.name)
            return
        if isinstance(stmt, ast.ClassDef):
            self._block(stmt.body, dict(names), dict(keys), func=func)
            return
        if isinstance(stmt, ast.Assign):
            self._check_calls(stmt.value, names, keys)
            origins = self._origins_in(stmt.value, names, keys)
            self._assign_targets(stmt.targets, origins, names, keys, stmt)
            return
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            self._check_calls(stmt.value, names, keys)
            origins = self._origins_in(stmt.value, names, keys)
            self._assign_targets([stmt.target], origins, names, keys, stmt)
            return
        if isinstance(stmt, ast.AugAssign):
            self._check_calls(stmt.value, names, keys)
            origins = self._origins_in(stmt.value, names, keys)
            if origins:
                self._assign_targets([stmt.target], origins, names, keys, stmt)
            return
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            self._check_calls(stmt.value, names, keys)
            origins = self._origins_in(stmt.value, names, keys)
            if self._returns is not None:
                self._returns.extend(origins)
            if (
                origins
                and func
                and (
                    func in ENTRY_FUNCTIONS
                    or func in self.ctx.entry_points
                    or SIGNAL_NAME_RE.search(func) is not None
                )
            ):
                self._sink("signal", stmt, origins, f"returned from '{func}()'")
            return
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            loop_names = {n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name)}
            self.loop_vars.append(loop_names)
            self._block(stmt.body, names, keys, func)
            self.loop_vars.pop()
            self._block(stmt.orelse, names, keys, func)
            return
        if isinstance(stmt, ast.While):
            self.loop_vars.append(names_in(stmt.test))
            self._block(stmt.body, names, keys, func)
            self.loop_vars.pop()
            self._block(stmt.orelse, names, keys, func)
            return
        if isinstance(stmt, ast.If):
            self._block(stmt.body, names, keys, func)
            self._block(stmt.orelse, names, keys, func)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            self._block(stmt.body, names, keys, func)
            return
        if isinstance(stmt, ast.Try):
            self._block(stmt.body, names, keys, func)
            for h in stmt.handlers:
                self._block(h.body, names, keys, func)
            self._block(stmt.orelse, names, keys, func)
            self._block(stmt.finalbody, names, keys, func)
            return
        if isinstance(stmt, ast.Expr):
            self._check_calls(stmt.value, names, keys)
            # Evaluate for sources so standalone calls (e.g. plotting) register origins.
            self._origins_in(stmt.value, names, keys)

    _returns: list[TaintOrigin] | None = None

    # ------------------------------------------------------------ summaries
    def summarize(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionSummary:
        """Run the body of ``fn`` with clean parameters and record what it returns/sets."""
        self._returns = []
        keys: dict[str, list[TaintOrigin]] = {}
        self._block(fn.body, {}, keys, func=fn.name)
        summary = FunctionSummary(fn.name, list(self._returns), dict(keys))
        self._returns = None
        return summary


def compute_summaries(ctx: ModuleContext, rounds: int = 3) -> dict[str, FunctionSummary]:
    """Fixed-point summaries for every function defined in the file (helpers of helpers)."""
    funcs = [
        n for n in ast.walk(ctx.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    summaries: dict[str, FunctionSummary] = {}
    for _ in range(rounds):
        new: dict[str, FunctionSummary] = {}
        for fn in funcs:
            analysis = _TaintAnalysis(ctx, record=False, summaries=summaries)
            s = analysis.summarize(fn)
            if s.returns or s.mutated_keys:
                new[fn.name] = s
        if {k: [id(o.node) for o in v.returns] for k, v in new.items()} == {
            k: [id(o.node) for o in v.returns] for k, v in summaries.items()
        } and set(new) == set(summaries):
            summaries = new
            break
        summaries = new
    return summaries


# --------------------------------------------------------------------- running
def analyze_source(
    source: str,
    filename: str = "<string>",
    config: StaticConfig | None = None,
    *,
    include_passes: bool = True,
    entry_points: frozenset[str] | set[str] | None = None,
) -> list[Finding]:
    """Run every registered static rule on ``source``.

    ``entry_points`` names functions whose return value is the live signal, in addition to
    the conventional names (``generate_signals`` …).
    """
    from quantproof.analyzers.static.rules import RULES

    cfg = config or StaticConfig()
    try:
        ctx = build_context(source, filename, cfg, entry_points=entry_points)
    except SyntaxError as exc:
        return [
            Finding(
                id="QP000",
                category=Category.STATIC,
                severity=Severity.FAIL,
                title="Source could not be parsed",
                message=f"SyntaxError: {exc.msg}",
                location=Location(file=filename, line=exc.lineno, column=exc.offset),
                why_it_matters="Code that does not parse cannot be audited or executed.",
                recommendation="Fix the syntax error and re-run the audit.",
            )
        ]
    findings: list[Finding] = []
    disabled = set(cfg.disabled_rules)
    for rule in RULES:
        if rule.id in disabled:
            continue
        raw = rule.check(ctx)
        kept: list[Finding] = []
        for f in raw:
            line = f.location.line if f.location else None
            if f.is_issue and ctx.suppressed(line, f.id):
                findings.append(
                    f.model_copy(
                        update={
                            "severity": Severity.INFO,
                            "title": f"{f.title} (suppressed)",
                            "message": f"Suppressed by inline comment: {f.message}",
                        }
                    )
                )
            else:
                kept.append(f)
        if kept:
            findings.extend(kept)
        elif include_passes and not any(f.id == rule.id for f in findings):
            findings.append(rule.passed(filename))
    return findings


def analyze_file(
    path: str | Path, config: StaticConfig | None = None, *, include_passes: bool = True
) -> list[Finding]:
    """Read a Python file and run :func:`analyze_source` on it."""
    p = Path(path)
    source = p.read_text(encoding="utf-8")
    return analyze_source(source, str(p), config, include_passes=include_passes)


def analyze_path(
    path: str | Path, config: StaticConfig | None = None, *, include_passes: bool = False
) -> list[Finding]:
    """Analyze a file or every ``*.py`` file under a directory."""
    p = Path(path)
    if p.is_file():
        return analyze_file(p, config, include_passes=include_passes)
    findings: list[Finding] = []
    for file in sorted(p.rglob("*.py")):
        if any(
            part.startswith(".") or part in {"__pycache__", "venv", ".venv"} for part in file.parts
        ):
            continue
        findings.extend(analyze_file(file, config, include_passes=include_passes))
    return findings
