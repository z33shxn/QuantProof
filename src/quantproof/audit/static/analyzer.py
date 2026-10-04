"""AST-based static analysis of Python research code.

The analyzer parses source code once into a :class:`ModuleContext` that exposes
pre-computed facts (resolved call names, scopes, a future-information taint
analysis, split events, ...). Each rule in :mod:`quantproof.audit.static.rules`
inspects that context and emits findings.

Static analysis is heuristic by nature: it reasons about syntax, not about the
values flowing at runtime. Findings therefore carry a confidence level, and the
runtime future-perturbation test (:mod:`quantproof.audit.causal`) provides
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

from quantproof.audit.models import Category, Finding, Location
from quantproof.audit.severity import Severity
from quantproof.config import StaticConfig

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


def is_negative_expr(node: ast.AST | None) -> tuple[bool, bool]:
    """Return ``(is_negative, certain)`` for a shift-period expression.

    ``shift(-1)`` → (True, True); ``shift(-n)`` → (True, False); ``shift(1)`` → (False, True).
    """
    if node is None:
        return False, True
    value = literal(node)
    if value is not NOT_LITERAL:
        return isinstance(value, (int, float)) and value < 0, True
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
    config: StaticConfig = field(default_factory=StaticConfig)

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
    source: str, filename: str = "<string>", config: StaticConfig | None = None
) -> ModuleContext:
    """Parse source and pre-compute facts. Raises :class:`SyntaxError` on invalid code."""
    tree = ast.parse(source, filename=filename)
    ctx = ModuleContext(
        source=source,
        filename=filename,
        tree=tree,
        lines=source.splitlines(),
        config=config or StaticConfig(),
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
class _TaintAnalysis:
    """Flow-insensitive-within-statement, order-sensitive taint tracking.

    Sources: ``shift``/``diff``/``pct_change`` with negative periods (QP001),
    ``np.roll`` and ``x[i + k]`` look-ahead indexing inside loops (QP003).
    Taint propagates through assignments to names and to string column keys.
    Sinks: values returned from strategy entry functions or assigned to
    signal/position names ("signal"), first argument of fit/predict ("feature").
    """

    def __init__(self, ctx: ModuleContext) -> None:
        self.ctx = ctx
        self.loop_vars: list[set[str]] = []

    def run(self) -> None:
        self._block(self.ctx.tree.body, {}, {}, func=None)

    # taint maps: identifier -> origins
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
            elif isinstance(sub, ast.Subscript):
                origin = self._lookahead_index(sub)
                if origin:
                    found.append(origin)
                key = literal(sub.slice)
                if isinstance(key, str) and key in keys:
                    found.extend(keys[key])
                elif isinstance(key, (list, tuple)):
                    for k in key:
                        if isinstance(k, str) and k in keys:
                            found.extend(keys[k])
                elif isinstance(sub.slice, ast.Name) and sub.slice.id in self.ctx.module_constants:
                    const = self.ctx.module_constants[sub.slice.id]
                    items = const if isinstance(const, (list, tuple)) else [const]
                    for k in items:
                        if isinstance(k, str) and k in keys:
                            found.extend(keys[k])
            elif isinstance(sub, ast.Name) and sub.id in names:
                found.extend(names[sub.id])
            elif isinstance(sub, ast.Attribute) and sub.attr in keys:
                found.extend(keys[sub.attr])
        # de-duplicate by node identity, keep order
        seen: set[int] = set()
        unique = []
        for o in found:
            if id(o.node) not in seen:
                seen.add(id(o.node))
                unique.append(o)
        return unique

    def _source_call(self, call: ast.Call) -> TaintOrigin | None:
        name, qual, _ = _resolve_call(call, self.ctx.imports)
        if name in SHIFT_LIKE:
            period = call.args[0] if call.args else get_kwarg(call, "periods")
            neg, _certain = is_negative_expr(period)
            if neg:
                return self._origin("QP001", call, f"{name}() with a negative period")
        if name == "roll" and qual.startswith(("numpy", "np")):
            shift = call.args[1] if len(call.args) > 1 else get_kwarg(call, "shift")
            neg, _ = is_negative_expr(shift)
            if neg:
                return self._origin("QP003", call, "np.roll with a negative shift")
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
                self.ctx.sinks.append(
                    TaintSink("signal", stmt, origins, f"assigned to signal/position '{label}'")
                )
            elif origins and label and LABEL_NAME_RE.match(label):
                self.ctx.sinks.append(TaintSink("label", stmt, origins, f"assigned to '{label}'"))

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
            if name in FIT_METHODS | PREDICT_METHODS and sub.args:
                origins = self._origins_in(sub.args[0], names, keys)
                if origins:
                    self.ctx.sinks.append(
                        TaintSink("feature", sub, origins, f"used as features in .{name}()")
                    )
                if name in FIT_METHODS and len(sub.args) > 1:
                    y_origins = self._origins_in(sub.args[1], names, keys)
                    if y_origins:
                        self.ctx.sinks.append(
                            TaintSink("label", sub, y_origins, f"used as target in .{name}()")
                        )
            elif name == "assign":
                for kw in sub.keywords:
                    if kw.arg is None:
                        continue
                    origins = self._origins_in(kw.value, names, keys)
                    if origins:
                        keys[kw.arg] = origins
                        if SIGNAL_NAME_RE.search(kw.arg):
                            self.ctx.sinks.append(
                                TaintSink("signal", sub, origins, f"assigned to '{kw.arg}'")
                            )

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
            # Function bodies see module-level taint but keep their own locals.
            self._block(stmt.body, dict(names), keys, func=stmt.name)
            return
        if isinstance(stmt, ast.ClassDef):
            self._block(stmt.body, dict(names), keys, func=func)
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
            if (
                origins
                and func
                and (func in ENTRY_FUNCTIONS or SIGNAL_NAME_RE.search(func) is not None)
            ):
                self.ctx.sinks.append(
                    TaintSink("signal", stmt, origins, f"returned from '{func}()'")
                )
            return
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            loop_names = {n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name)}
            self.loop_vars.append(loop_names)
            self._block(stmt.body, names, keys, func)
            self.loop_vars.pop()
            self._block(stmt.orelse, names, keys, func)
            return
        if isinstance(stmt, ast.While):
            loop_names = names_in(stmt.test)
            self.loop_vars.append(loop_names)
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


# --------------------------------------------------------------------- running
def analyze_source(
    source: str,
    filename: str = "<string>",
    config: StaticConfig | None = None,
    *,
    include_passes: bool = True,
) -> list[Finding]:
    """Run every registered static rule on ``source``."""
    from quantproof.audit.static.rules import RULES

    cfg = config or StaticConfig()
    try:
        ctx = build_context(source, filename, cfg)
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
