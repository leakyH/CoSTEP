# CoSTEP / CoSTEP-IR

CoSTEP (Consistency-Constrained Spatiotemporal Epidemic Prediction) combines a
graph neural encoder and a mechanistic decoder with cross-origin consistency
constraints. CoSTEP-IR extends it to irregularly reported observations.

## Install

Python 3.10+.

```bash
git clone https://github.com/leakyH/CoSTEP.git
cd CoSTEP
python -m pip install .
```

## Experiments

The package includes JAPAN, US_STATE, and JAPAN-Irregular data. Experiments use
14 days of observations to forecast the next 14 days. The bundled
[recipes](costep/recipes/formal_10seeds.json) run seeds 42–51.

| Profile (`--profiles`) | Experiment | Variants (`--variants`) |
| --- | --- | --- |
| `japan_regular_costep` | JAPAN main experiment | `full` |
| `us811_ir_main` | US_STATE main experiment (8:1:1 split) | `full` |
| `japan_ablation` | JAPAN ablations | `no_cons`, `mepognn` |
| `us811_ablation` | US_STATE ablations | `no_ir`, `no_cons`, `mepognn` |
| `japan_irregular_paper` | Irregular reporting, levels L0–L4 | `costep`, `costep_ir` |

`no_ir` removes the interval-aware representation; `no_cons` removes consistency
constraints; `mepognn` runs the MepoGNN architecture ablation for each dataset.
**`no_adaptation` is the Base model**: use the Base results from the corresponding
experiment, or select `--stage base` to run only Base.

## Run

For a single experiment:

```bash
python -m costep train --dataset japan --output outputs/japan
```

Use `--dataset us` for US_STATE or `--dataset japan_irregular --level L4` for
JAPAN-Irregular at L4. Training includes Base, Post adaptation, and evaluation;
use `--no-post` for Base only. Use a new output directory for each run.

For a ten-seed experiment, select a profile from the table and run:

```bash
python -m costep.launcher plan --profiles japan_ablation
python -m costep.launcher train --profiles japan_ablation --device cuda:0
python -m costep.launcher evaluate --profiles japan_ablation --device cuda:0
python -m costep.launcher summary --profiles japan_ablation --save outputs/japan_ablation_summary.json
```

Use `--variants no_cons` to select one variant, `--device cpu` for CPU execution,
and `--output-root` to change the default `outputs/reproduction` directory.
Summary reports MAE and reportMAE means and sample standard deviations across seeds.
See `python -m costep --help` and `python -m costep.launcher --help` for options.

[MIT License](LICENSE). [Data sources and third-party notices](licenses/README.md).
