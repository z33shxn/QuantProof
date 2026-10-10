# Verdict rules

The overall verdict is derived from findings by three ordered rules
(`quantproof.results.determine_verdict`):

1. **FAIL** if at least one finding has severity FAIL: a critical research-validity
   violation was established.
2. **WARN** if no finding is FAIL and at least one is WARN: an important methodological
   weakness exists.
3. **PASS** if every executed check produced only PASS or INFO findings.

There is no score. Each analyzer documents the thresholds that decide its findings'
severities (module docstrings and the methodology pages), records them in the finding's
`evidence`, and the report lists exactly which findings drove the verdict.

Severity policy:

- **FAIL** is reserved for established validity violations: future information reaching
  decisions or model inputs (static taint, runtime perturbation, value-level leakage),
  impossible data, trades before their signals, and headline performance that depends on
  unrealistic execution assumptions.
- **WARN** marks weak or context-dependent evidence: statistical insignificance after
  multiple testing, high PBO, weak OOS results, fragile optima, missing costs, shuffled
  splits.
- **INFO** records context (trial counts, suppressed findings, checks not run).

Statistical findings never produce FAIL on their own: weak evidence is not proof of
invalid research.

A PASS verdict is not a recommendation to trade.
