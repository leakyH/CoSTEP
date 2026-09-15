# CoSTEP / CoSTEP-IR

## Installation

Python 3.10+, PyTorch 2.8.0, NumPy 2.2.6, pandas 2.3.3.

```bash
python -m pip install .
```

## Experiments

Configurations are in `costep/recipes/formal_10seeds.json`. The seed list is
42–51. Each profile contains shared training settings and experiment variants.

```bash
python -m costep.launcher list
python -m costep.launcher plan --profiles japan_regular_costep --save plan.json
python -m costep.launcher train --profiles japan_regular_costep --device cpu
python -m costep.launcher evaluate --profiles japan_regular_costep
python -m costep.launcher summary --profiles japan_regular_costep --save summary.json
```

Use `--matrix path/to/config.json` for a modified configuration and
`--output-root path/to/results` to set the result directory.
Keep the same configuration and output directory for training and evaluation.

Variants have three settings:

- `enabled`: include this variant in the run.
- `overrides`: change `curve`, `ir`, or `consistency`.
- `stages`: `["base", "post"]` or `["base"]` for no-Post results.

`curve: false` uses independent daily beta/gamma outputs. `consistency: false`
disables both CAPC and SRC. Select variants with `--variants no_curve no_curve_no_cons`.
The `full` variant reuses the corresponding main experiment. Native/norm variants
are not implemented.

Training runs sequentially. Use `--stage base` or `--stage post` to run one stage.
Post uses the matching Base checkpoint. `--skip-existing` skips completed stages
with matching configurations; interrupted runs need a new output directory.

Evaluation saves predictions and metrics. `score` and `summary` recalculate
metrics from saved predictions. Summaries report the mean and sample standard
deviation across seeds. Use `--allow-partial` to summarize an incomplete seed list.
For one seed, standard deviation is null.

Irregular experiments pair model and calendar seeds. They use observation_stock
with STE, 500 Base and 500 Post epochs, and Post selection from epoch 100.

## Single runs

```bash
python -m costep train --dataset japan --curriculum none --output outputs/japan
python -m costep adapt --checkpoint outputs/japan/base.pt --output outputs/post
python -m costep evaluate --checkpoint outputs/post/post.pt --output outputs/evaluation
python -m costep score --predictions outputs/evaluation/post_predictions.npz
```

Single-run training includes Post by default; `--no-post` runs Base only.
Use a new output directory for each command.

## Data

Japan retains the MepoGNN array format and node order. Region labels JP-01 to
JP-47 were added by us and checked against external population information;
the original MepoGNN arrays do not contain these labels.

US observations are stored as compressed CSV with dates and region IDs. Flow
matrices remain NumPy arrays. The 51-node order follows the source reader and
excludes FIPS 44. Irregular reference calendars cover L0–L4 and seeds 42–51.

Japan uses daily peakMAE. US and Japan-Irregular use reporting-interval peakMAE.
Predictions include node IDs, forecast origins, and reporting metadata.

## US data sources

- Cases: [JHU CSSE COVID-19 Data](https://github.com/CSSEGISandData/COVID-19),
  Copyright Johns Hopkins University 2020, distributed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  The packaged table contains the selected date range and regions, aligned to
  the model's daily data grid. For publications, cite Dong, Du and Gardner,
  *An interactive web-based dashboard to track COVID-19 in real time*,
  Lancet Infectious Diseases 20(5):533–534 (2020),
  https://doi.org/10.1016/S1473-3099(20)30120-1.
- Demographic node inputs: [U.S. Census Bureau ACS five-year data](https://www.census.gov/newsroom/press-kits/2021/acs-5-year.html).
  Region-level inputs are aligned using the region IDs in the packaged table.
  Census Bureau statistical data are public-domain U.S. government information.
- Mobility: the GeoDS COVID19USFlows dataset credited below, aligned to the same
  node order. The source-specific data terms remain separate from the code's MIT license.

## Acknowledgments and licenses

CoSTEP is released under the MIT License. Copyright (c) 2026 Shaoyu Huang.
Third-party notices are included in `licenses/`.

- [MepoGNN](https://github.com/deepkashiwa20/MepoGNN): source research code and
  packaged JAPAN data. The upstream MIT copyright and permission notice are
  retained in `licenses/MepoGNN_LICENSE`.
- [GeoDS COVID19USFlows](https://github.com/GeoDS/COVID19USFlows): the published
  mobility dataset used to construct the US flows, released under MIT. Its notice
  is retained in `licenses/COVID19USFlows_LICENSE`. This attribution refers to
  that published dataset, not to all SafeGraph products.
- [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet): portions of the graph
  convolution and temporal convolution code are adapted from this work. Its
  copyright and permission notice are retained in `licenses/GraphWaveNet_LICENSE`.

For COVID19USFlows, please cite Kang et al., *Multiscale dynamic human mobility
flow dataset in the U.S. during the COVID-19 epidemic*, Scientific Data 7, 390
(2020), https://doi.org/10.1038/s41597-020-00734-5.

