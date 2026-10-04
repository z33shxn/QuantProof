# Command line

```text
quantproof [--debug] [--version] COMMAND …

quantproof audit      --strategy/-s FILE --data/-d FILE [--benchmark FILE] [--config FILE]
                      [--output/-o FILE] [--format html|json|markdown|text]
                      [--profile quick|standard|strict] [--quick] [--seed N]
                      [--timestamp-column NAME] [--symbol-column NAME]
                      [--fail-on warn|fail|never] [--quiet] [--verbose]
quantproof audit      --returns FILE [--trials FILE] [--data FILE] …        # results mode
quantproof report     RESULT.json [--format html|markdown|json|text] [--output FILE]
quantproof validate   DATA [--config FILE] [--timestamp-column NAME] [--symbol-column NAME]
                      [--fail-on …]
quantproof scan       PATH [--show-passes] [--fail-on …]
quantproof rules      [--category CATEGORY] [--json] [--markdown]
quantproof rules show RULE_ID [--json]
quantproof statistics RETURNS [--column C] [--periods 252] [--trials N] [--benchmark-sharpe S]
                      [--risk-free R] [--bootstrap B] [--seed N] [--json]
quantproof init-config [quantproof.yaml] [--force]
quantproof generate-data OUTPUT.csv|parquet [--n 1500] [--seed 2026] [--ar1 -0.15]
quantproof version
```

Every command has `--help` with examples.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Completed; overall status PASS/INFO, or below the `--fail-on` level |
| 1 | Completed; overall status WARN (with `--fail-on warn`, the default) |
| 2 | Completed; overall status FAIL (unless `--fail-on never`) |
| 3 | Invalid input: unknown option, unreadable file, unusable data or strategy, unknown rule id. Nothing was audited |
| 4 | Internal error (a bug in QuantProof). Re-run with `quantproof --debug …` for a traceback |

`--fail-on fail` maps WARN to 0 (useful when warnings should not break CI);
`--fail-on never` always exits 0 after a completed audit. Errors never produce a raw Python
traceback unless `--debug` is given.

## Notes

- The report format is inferred from the output extension (`.html`, `.json`, `.md`,
  `.txt`). `--format markdown` without `--output` prints Markdown to stdout (for PR
  comments).
- `--verbose` prints every finding with why it matters, potential impact, how to
  investigate and the recommendation.
- `quantproof rules show QP001` prints a rule's severity policy, rationale, limitations
  and example; `quantproof rules --json` is the machine-readable registry.
- Multi-asset data: long format with a `symbol` (or `ticker`, `asset` …) column is
  detected automatically; pass `--symbol-column` otherwise.
