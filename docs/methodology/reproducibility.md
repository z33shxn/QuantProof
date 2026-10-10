# Reproducibility

## Problem

An audit is only useful if someone else can reproduce it: same code, same data, same
configuration, same environment, same random seed.

## Implementation

Every audit returns a manifest (`result.manifest`, YAML in the HTML/Markdown reports, JSON
under `reproducibility`):

```yaml
quantproof_version: 0.1.0
strategy: {name: dual_ma_trend, path: …/strategy.py, sha256: …}
data:
  source: examples/data/prices.parquet
  file_sha256: …
  content_sha256: …
  schema: {rows: 1500, columns: {open: float64, …}, start: …, end: …, timezone: null}
parameters: {defaults: {…}, grid: {…}, grid_size: 16}
validation: {walk_forward: {…}, pbo: {…}, cpcv: {groups: 6, test_groups: 2, embargo: 0.01}}
execution: {signal_lag: 1, commission_bps: 1.0, …, declared_by_strategy: {…}}
configuration: {sha256: …, values: {…}}
experiment: {random_seed: 42, git: {commit: …, dirty: false}}
environment: {python: 3.12.x, platform: …, packages: {numpy: …, pandas: …}}
lineage: [{step: load_and_prepare_data, inputs: {raw: …}, outputs: {prices: …}}, …]
content_hash: …
created_at: 2026-…Z
```

**Data hashing** (`hash_dataframe`) is defined explicitly rather than via pickle/Parquet
bytes (which embed library versions): column names, a dtype tag, and values. Floats are hashed as
little-endian float64 with NaNs canonicalized, integers as int64, datetimes as UTC
nanoseconds (independent of resolution and display timezone), and other values as UTF-8
strings. The index is hashed the same way.

**`content_hash`** covers every manifest field except `created_at`, so identical inputs
give identical hashes. The integration tests run the clean example twice and compare both
the content hash and the full JSON result (minus the timestamp).

**Determinism.** Every random procedure (bootstrap, CSCV sub-sampling, grid sub-sampling,
perturbations, Reality Check) takes the configuration seed; nothing uses global random
state.

## Limitations

Results are bit-identical for the same environment. Across NumPy/SciPy versions,
floating-point results can differ in the last digits; the environment section records the
versions used. Strategy code that reads external state is outside QuantProof's control (the
non-determinism check in the causality test flags unseeded randomness).
