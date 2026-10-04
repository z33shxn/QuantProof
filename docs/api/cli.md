# Command line

```text
quantproof version
quantproof audit      --strategy/-s FILE --data/-d FILE [--benchmark FILE] [--config FILE]
                      [--output/-o FILE] [--format html|json|markdown|text]
                      [--quick] [--seed N] [--fail-on fail|warn|never] [--quiet] [--verbose]
quantproof audit      --returns FILE [--trials FILE] [--data FILE] …        # results mode
quantproof report     RESULT.json [--format html|markdown|json|text] [--output FILE]
quantproof validate   DATA [--config FILE] [--timestamp-column NAME] [--fail-on …]
quantproof scan       PATH [--show-passes] [--fail-on …]
quantproof rules
quantproof statistics RETURNS [--column C] [--periods 252] [--trials N] [--benchmark-sharpe S]
                      [--risk-free R] [--bootstrap B] [--seed N] [--json]
quantproof init-config [quantproof.yaml] [--force]
quantproof generate-data OUTPUT.csv|parquet [--n 1500] [--seed 2026] [--ar1 -0.15]
```

- The report format is inferred from the output extension (`.html`, `.json`, `.md`,
  `.txt`). `--format markdown` without `--output` prints Markdown to stdout (for PR
  comments).
- Exit codes: `0` success with a verdict below `--fail-on` (default `fail`), `1` verdict at
  or above it, `2` usage or input error (message on stderr).
