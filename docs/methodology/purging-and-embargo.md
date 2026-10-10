# Purging and embargo

## Problem

Labels in finance often span time: a 5-day forward return computed at bar *i* uses
prices up to *i + 5*. If a training observation's label overlaps the test period, the
model is trained on information from the test set. Separately, serial correlation means
observations just after the test set are informative about it.

## Sample-time vs event-time

Folds are always contiguous blocks of **observations** (sample-time splitting). What
changes is how label overlap is measured:

| Mode | Argument | Interval of observation *i* | Use when |
|---|---|---|---|
| sample-time | `label_horizon=h` | `[i, i + h]` in bar positions | every label spans the same number of bars |
| event-time | `event_end=` (alias `t1=`), optional `event_start=` | `[event_start_i, event_end_i]` in time | labels have variable length (triple-barrier, event studies) |

Giving both `label_horizon` and `event_end` is an error.

## Implementation

**Label intervals** (`label_intervals`). Each observation *i* has an interval
`[start_i, end_i]`. Event-time intervals are converted to nanoseconds since the epoch in
UTC, independent of the index's datetime resolution and timezone. Input validation
(each raises `QuantProofInputError`):

- exactly one end time per observation, no missing (NaT) end times;
- an `event_end` Series must be indexed exactly like the data (same timestamps, same order);
- start and end must both be timezone-aware or both naive (aware values in different
  timezones are compared in UTC);
- observation start times must be non-decreasing; unsorted data is rejected rather than
  silently mis-purged;
- every end is ≥ its start.

**Purging** (`purge`). For each *contiguous* test block `[a, b]`, the test span is
`[start_a, max(end_a … end_b)]`. A training observation *j* is removed if
`start_j ≤ span_end` and `end_j ≥ span_start` (its label interval overlaps the span).
This removes observations both before the test block (labels extending into it) and
inside the span after it.

**Embargo** (`apply_embargo`, `embargo_size`). `h = ceil(embargo · n)` for a fraction,
`h` bars for an integer, or (with event-time labels only) a duration such as `"5D"` /
`pd.Timedelta(days=5)`: training observations whose start lies in
`(span_end, span_end + duration]` are removed. Following López de Prado (2018, snippet 7.3), the embargo window
starts **after the end of the test block's label span**: with `T` the latest label end in
the block, the anchor is the last position whose observation time is ≤ `T`, and training
positions in `(anchor, anchor + h]` are removed. Purging and embargo are therefore
additive. Without label intervals the anchor is the last test position.

If purging and embargo remove every training observation of a split, the splitter raises
an error naming the split instead of yielding an empty training set.

**Purged K-fold** (`PurgedKFold(n_splits, event_end=None, event_start=None, t1=None,
label_horizon=0, embargo=0)`): contiguous folds, each purged and embargoed.
scikit-learn compatible (`split`, `get_n_splits`).

## Boundary conventions (asserted by `tests/unit/test_temporal_boundaries.py`)

```
positions      0  1  2  3 | 4  5  6  7 | 8  9 10 11      label_horizon = 2, embargo = 1
fold 0 test    T  T  T  T                                test span = [0, 3 + 2] = [0, 5]
fold 0 train               P  P  E  .   .  .  .  .       P = purged (start ≤ 5), E = embargo
fold 1 test                T  T  T  T                    span = [4, 9]
fold 1 train   .  .  P  P               P  P  E  .       2: [2,4] touches 4 → purged
```

- Overlap is tested with **inclusive** ends: a label ending exactly at the test start is
  purged.
- The embargo starts after the *label span*, not after the last test bar.
- These splits are checked against an independent brute-force implementation
  (pairwise interval intersection) for PurgedKFold and every CPCV split.

```python
from quantproof.validation import PurgedKFold

for train, test in PurgedKFold(5, label_horizon=5, embargo=0.01).split(X):
    model.fit(X.iloc[train], y.iloc[train])
```

## Assumptions

- Event end times are at or after their start times (validated).
- Embargo applies after test blocks only; observations before a test block are handled by
  purging.

## Example

With 100 observations, `label_horizon=3`, embargo 2 bars and test fold 0–19: training
observations 0–22 overlap the test span `[0, 22]` and are purged; the embargo then removes
23–24; training resumes at 25 (asserted in the tests).

## Limitations

Embargo size is a judgement about how long dependence lasts; QuantProof does not estimate
it. Purging requires correct label end times.

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7 (sections
  7.4.1–7.4.3, snippets 7.1–7.3).
