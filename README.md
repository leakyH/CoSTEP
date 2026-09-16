# CoSTEP / CoSTEP-IR

CoSTEP (Consistency-Constrained Spatiotemporal Epidemic Prediction) is a hybrid
neural–mechanistic model for regional epidemic forecasting. It combines a graph
neural encoder and a mechanistic decoder with cross-origin consistency constraints.
CoSTEP-IR extends it to observations reported as totals over irregular intervals.

## Install

Python 3.10+.

```bash
git clone https://github.com/leakyH/CoSTEP.git
cd CoSTEP
python -m pip install .
```

## Run

```bash
python -m costep train --dataset japan --output outputs/japan
```

Use `--dataset us` or `--dataset japan_irregular` for the other datasets.
Training includes Post and evaluation; use `--no-post` for Base only.
Choose a new output directory for each run. See `python -m costep --help` for options.

To run a configuration from `costep/recipes/formal_10seeds.json`:

```bash
python -m costep.launcher list
python -m costep.launcher train --profiles japan_regular_costep --device cpu
```

[MIT License](LICENSE). [Data sources and third-party notices](licenses/README.md).
