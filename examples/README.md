# Examples

Deliberately constructed strategies used to show — and test — what QuantProof detects.
See [docs/examples/README.md](../docs/examples/README.md) for the full walkthrough and
results.

| Directory | Purpose |
|---|---|
| `clean_strategy/` | causal strategy with declared, realistic assumptions (no false positives expected) |
| `lookahead_strategy/` | deliberate look-ahead (`shift(-1)`, centered window) |
| `leakage_strategy/` | deliberate ML leakage (global scaler, shuffled split, future feature); needs scikit-learn |
| `overfit_strategy/` | best of a 240-point grid on a random walk |
| `unrealistic_execution/` | same-bar fills and zero costs |
| `data/` | synthetic datasets and the script that generates them |
| `notebooks/` | interactive walkthrough |

```bash
pip install -e ".[all]"
python examples/run_examples.py
quantproof audit -s examples/lookahead_strategy/strategy.py -d examples/data/prices.parquet -o report.html
```
