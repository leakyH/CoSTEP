# Data sources and third-party notices

CoSTEP is released under the MIT License. Copyright (c) 2026 Shaoyu Huang.
The source-specific data terms remain separate from the code's MIT license.

- [MepoGNN](https://github.com/deepkashiwa20/MepoGNN): a portion of code and
  packaged JAPAN data are adapted from this work. The upstream MIT notice is retained in [MepoGNN_LICENSE](MepoGNN_LICENSE).
- [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet): a portion of the graph
  convolution and temporal convolution code in MepoGNN is adapted from this work,
  and thus our work is built on this work. Its notice is retained in
  [GraphWaveNet_LICENSE](GraphWaveNet_LICENSE).
- [GeoDS COVID19USFlows](https://github.com/GeoDS/COVID19USFlows): the published
  mobility dataset used to construct the US flows, aligned to the model's node
  order. Its MIT notice is retained in [COVID19USFlows_LICENSE](COVID19USFlows_LICENSE).
  This attribution refers to that published dataset, not to all SafeGraph products.
  Please cite Kang et al., *Multiscale dynamic human mobility flow dataset in the
  U.S. during the COVID-19 epidemic*, Scientific Data 7, 390 (2020),
  https://doi.org/10.1038/s41597-020-00734-5.
- [JHU CSSE COVID-19 Data](https://github.com/CSSEGISandData/COVID-19): US cases,
  Copyright Johns Hopkins University 2020, distributed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  The packaged table contains the selected date range and regions, aligned to
  the model's daily data grid. For publications, cite Dong, Du and Gardner,
  *An interactive web-based dashboard to track COVID-19 in real time*,
  Lancet Infectious Diseases 20(5):533–534 (2020),
  https://doi.org/10.1016/S1473-3099(20)30120-1.
- [U.S. Census Bureau ACS five-year data](https://www.census.gov/newsroom/press-kits/2021/acs-5-year.html):
  demographic node inputs, aligned using the region IDs in the packaged table.
  Census Bureau statistical data are public-domain U.S. government information.
