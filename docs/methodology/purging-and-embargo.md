# Purging and embargo

## Problem

Labels in finance often span time: a 5-day forward return computed at bar *i* uses
prices up to *i + 5*. If a training observation's label overlaps the test period, the
model is trained on information from the test set. Separately, serial correlation means
observations just after the test set are informative about it.

## Implementation

**Label intervals** (`label_intervals`). Each observation *i* has an interval
`[start_i, end_i]`:

- with `t1` (a Series mapping each observation time to the time its label is known, e.g.
  a triple-barrier touch time), intervals are in nanoseconds since the epoch, independent
  of the index's datetime resolution;
- otherwise `[i, i + label_horizon]` in bar positions.

**Purging** (`purge`). For each *contiguous* test block `[a, b]`, the test span is
`[start_a, max(end_a … end_b)]`. A training observation *j* is removed if
`start_j ≤ span_end` and `end_j ≥ span_start` (its label interval overlaps the span).
This removes observations both before the test block (labels extending into it) and
inside the span after it.

**Embargo** (`apply_embargo`, `embargo_size`). `h = ceil(embargo · n)` for a fraction, or
`h` bars for an integer. Following López de Prado (2018, snippet 7.3), the embargo window
starts **after the end of the test block's label span**: with `T` the latest label end in
the block, the anchor is the last position whose observation time is ≤ `T`, and training
positions in `(anchor, anchor + h]` are removed. Purging and embargo are therefore
additive. Without label intervals the anchor is the last test position.

**Purged K-fold** (`PurgedKFold(n_splits, t1=None, label_horizon=0, embargo=0)`):
contiguous folds, each purged and embargoed. scikit-learn compatible (`split`,
`get_n_splits`).

```python
from quantproof.validation import PurgedKFold

for train, test in PurgedKFold(5, label_horizon=5, embargo=0.01).split(X):
    model.fit(X.iloc[train], y.iloc[train])
```

## Assumptions

- `t1` values are at or after their observation times (validated).
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
