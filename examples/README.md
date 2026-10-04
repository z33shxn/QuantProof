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
| `cross_sectional_momentum/` | realistic multi-asset (panel) research with 36 declared variants |
| `proof_of_value/` | flawed first draft → audit → fix → re-run (`python examples/proof_of_value/run.py`) |
| `data/` | synthetic datasets and the script that generates them |
| `notebooks/` | interactive walkthrough |

```bash
pip install -e ".[all]"
python examples/run_examples.py
python examples/proof_of_value/run.py
quantproof audit -s examples/lookahead_strategy/strategy.py -d examples/data/prices.parquet -o report.html
```
