# Security policy

## Execution boundary

**QuantProof executes strategy code.** `quantproof audit --strategy file.py`,
`quantproof.audit(strategy=...)` and `load_strategy` import the file with
`importlib`, which runs all of its top-level code, and then call `generate_signals`
repeatedly (including on perturbed data). This happens in the current Python process,
with the privileges of the user running QuantProof.

QuantProof does **not** provide a sandbox. Do not audit untrusted strategy code on a
machine that holds credentials, private data or network access you care about. For
third-party code, run QuantProof inside an isolated environment (a container or VM with no
secrets and restricted network, or a CI runner without repository secrets).

Code-free operations do **not** execute user code: static analysis (`quantproof scan`,
`analyze_file`, which parses source with `ast` without importing it), data validation
(`quantproof validate`), statistics, and results-mode audits (`returns=`, `artifacts=`).

Other boundaries:

- Configuration files are parsed with safe loaders (`yaml.safe_load`, `json`, `tomllib`).
  QuantProof never uses `pickle`; data is read only from CSV and Parquet (`pandas`/
  `pyarrow`), and unreadable files are reported as input errors.
- The only subprocess call is `git rev-parse HEAD` / `git status --porcelain` (argument
  list, no shell, 5 s timeout) in the strategy's directory, to record the commit in the
  manifest.
- File paths are used as given; QuantProof writes only the report/config paths you pass
  (creating parent directories) and never deletes files.
- HTML reports are rendered with Jinja2 autoescaping; the embedded JSON encodes every `<`
  as `\u003c`, so finding text cannot close the script element. Reports load no external
  resources.
- Reports and manifests include file paths, package versions, platform and the Git commit.
  Review them before sharing outside your organization.

## Reporting a vulnerability

Please report vulnerabilities privately, not in public issues: use the repository's
**Security** tab → **Report a vulnerability** (GitHub private vulnerability reporting). If
that option is not available, open a minimal public issue asking a maintainer for a private
contact channel, without any vulnerability details. Include steps to reproduce and the
affected version in the private report.

## Supported versions

Security fixes are made on the latest released minor version.
