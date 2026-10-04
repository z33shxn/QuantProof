# Contributing

Thank you for improving QuantProof. The bar for changes is the project's own standard:
**correct enough that a quant can inspect it, tested well enough that a reviewer can trust
it.**

## Setup

```bash
git clone https://github.com/z33shxn/QuantProof.git
cd QuantProof
uv sync --extra dev          # reproducible, from uv.lock
# or: pip install -e ".[dev]"
```

Checks (all run in CI):

```bash
pytest                                   # unit, integration, property-based, doctests
pytest --cov=quantproof --cov-fail-under=85
ruff check . && ruff format --check .
mypy
python -m build
```

After changing dependencies, run `uv lock` and commit `uv.lock`.

## Principles

- **No fake intelligence or hype.** Results come from deterministic or statistical
  analysis. Use "identifies", "provides evidence", never "proves" or "guarantees".
- **Quality over feature count.** A rule that cannot be implemented and tested properly
  does not ship; experimental features are labelled experimental.
- **Explicit verdicts.** Severities are decided by documented thresholds recorded in the
  finding's `evidence`.
- **Determinism.** All randomness takes the configuration seed.
- **Every bug gets a regression test** in `tests/unit/test_regressions.py` (or next to the
  affected module's tests).
- Do not weaken, skip or `xfail` tests to make CI green without a documented reason.

## Adding a static rule

Rule metadata lives in one place, the registry in `src/quantproof/rules.py`; rule
classes contain only detection logic.

1. Add an entry to `_STATIC` in `src/quantproof/rules.py` (id, name, severity policy,
   description, rationale, impact, how to investigate, remediation, limitations, example,
   pass title).
2. Add a class to `src/quantproof/analyzers/static/rules.py`:

   ```python
   @register
   class MyRule(StaticRule):
       id = "QP016"  # title, rationale and remediation come from the registry

       def check(self, ctx: ModuleContext) -> list[Finding]:
           return [
               self.finding(
                   ctx, call.node, Severity.WARN, "Specific message.", confidence=Confidence.MEDIUM
               )
               for call in ctx.calls_named("suspicious_function")
           ]
   ```

   For look-ahead patterns use `timing_finding(...)`, which classifies the use as a live
   decision (FAIL), label/analysis (INFO) or unknown (WARN).
3. Add a deliberately broken file `tests/fixtures/broken/qp016_*.py`, register it in
   `EXPECTED` in `tests/unit/test_static_rules.py`, and add adversarial and false-positive
   cases to `tests/unit/test_static_adversarial.py`.
4. Regenerate the catalogue: `quantproof rules --markdown > docs/api/rules.md` (a test
   fails when it is out of date). Update `docs/methodology/lookahead-detection.md` if the
   rule changes what the analyzer can resolve.

## Adding an adapter

Implement `quantproof.adapters.Adapter` (`name` and `load(source, **kwargs) ->
ResearchArtifacts`) in its own module, keep the engine import inside that module, add
tests with small synthetic engine outputs, and document gross vs net returns. See
`docs/api/adapters.md`.

## Adding or updating a cost provider

Jurisdiction-specific rates live in `src/quantproof/execution/providers/`. Every rate needs
an effective date range and a source; never overwrite a historical schedule — close it with
`effective_to` and add a new one. Update `last_reviewed` and the tests with hand-computed
values.

## Pull requests

Keep changes focused, update the CHANGELOG under "Unreleased", and describe how you
verified the change. By contributing you agree that your contributions are licensed under
the MIT License.
