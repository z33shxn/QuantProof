# Parameter sensitivity

## Problem

An optimum that sits on a plateau of similar results is more credible than a single spike
surrounded by poor neighbours; the spike is what a search over noise tends to find.

## Implementation

`quantproof.sensitivity.analyze_parameter_surface(results, params, metric, oos_metric=None)`
with one row per combination:

- grid coordinates are the ranks of each parameter's sorted unique values;
- **neighbours** are points at Chebyshev distance 1 (every parameter may move one step);
- **plateau ratio** = median(neighbour metric) / best metric (when best > 0);
- **spike z** = (best − mean of neighbours) / std of the surface;
- classification: robust (ratio ≥ 0.7), moderate, fragile (< 0.4), undetermined (best ≤ 0
  or no neighbours) — thresholds in `SensitivityConfig`;
- local maxima, share of the grid ≥ half the best value;
- with `oos_metric`: Spearman rank correlation of IS vs OOS across the grid, OOS value at
  the IS optimum, and whether the OOS optimum is the IS optimum.

In the audit, the metric is the full-sample net Sharpe of each `PARAM_GRID` combination
(QP-SENS-001), and IS/OOS are the first and second halves of the sample (QP-SENS-002 warns
if the rank correlation is negative). `render_ascii_surface` and the HTML heat-map show a
2-D slice with other parameters at their best values; ★ marks the maximum.

```text
fast\slow      40     50     60     70
10           0.20   0.20   0.20   0.20
15           0.20  3.00★   0.20   0.20
20           0.20   0.20   0.20   0.20
```

(an isolated spike — classified fragile.)

## Limitations

A narrow optimum is a warning, not a verdict: some genuine effects are narrow. Grids that
are sub-sampled (`validation.max_trials`) have missing neighbours.

## References

- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*, 2nd ed., ch. on
  parameter stability. Wiley.
- Bailey, D. H. et al. (2014). Pseudo-mathematics and financial charlatanism.
  *Notices of the AMS*, 61(5).
