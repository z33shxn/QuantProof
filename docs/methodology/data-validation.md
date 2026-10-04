# Data validation

## Problem

Backtests inherit every flaw in their input: unsorted rows break `shift`, duplicated bars
double-count returns, forward-filled prices invent fills, unadjusted splits create
spectacular "returns".

## Implementation

`quantproof.data.validate_data(raw_frame, DataValidationConfig)` runs on the **raw** table
(before cleaning), so problems are reported before `prepare_prices` fixes them. Every fix
(dropping invalid timestamps, sorting, de-duplicating) is recorded as an INFO finding
(`QP-DATA-000`); nothing is changed silently.

| Rule | Check | Severity |
|---|---|---|
| QP-DATA-001 | unparseable timestamps (rows dropped) | WARN |
| QP-DATA-002 | non-monotonic timestamps (message names the first offending pair) | WARN |
| QP-DATA-003 | missing price column / missing OHLC fields | FAIL / INFO (FAIL if `require_ohlc`) |
| QP-DATA-004 | duplicated timestamps, with affected range | WARN (INFO if `allow_duplicate_timestamps`) |
| QP-DATA-005 | fully duplicated observations | WARN |
| QP-DATA-006 | impossible OHLC (high < max(open, close), low > min(open, close), high < low) | FAIL |
| QP-DATA-007 | zero or negative prices | FAIL |
| QP-DATA-008 | missing values per column | WARN |
| QP-DATA-009 | gaps > `max_gap_multiple` × median spacing; intraday gaps across calendar days count as session breaks | WARN |
| QP-DATA-010 | mixed tz-aware/naive timestamps (WARN); multiple UTC offsets (INFO, e.g. DST) | WARN / INFO |
| QP-DATA-011 | runs of ≥ `max_stale_run` identical closes (forward fill) | WARN |
| QP-DATA-012 | infinite values | FAIL |
| QP-DATA-013 | single-period |return| > `extreme_return_threshold` | WARN |

Timestamps with explicit offsets are converted to UTC; naive timestamps are kept as one
consistent local time. All time arithmetic uses nanoseconds regardless of the index's
resolution (pandas may infer second or microsecond resolution).

Fatal problems raise `QuantProofDataError` with a fix, e.g.
`Could not find a timestamp column. Expected a DatetimeIndex or a column named one of: …`.

## Assumptions

Daily data: weekends and single holidays fall under the default 5× gap threshold.
Intraday data: overnight and weekend breaks are not gaps.

## Example

`quantproof validate examples/data/prices.csv` → 13 PASS findings.

## Limitations

Rules check internal consistency, not correctness against another source. Corporate
actions, survivorship and point-in-time availability cannot be verified from prices alone.
