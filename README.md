# PsyStat

**PsyStat** is a free, open-source statistical analysis application built for psychologists, researchers, and students. It provides a point-and-click interface for a comprehensive suite of analyses — from basic descriptives to structural equation modeling — with APA-formatted output and publication-ready charts.

---

## Features

PsyStat organises its tools into 17 analysis modules, accessible from the top navigation bar:

| Module | What it does |
|--------|-------------|
| **Data Management** | Import CSV/Excel, edit cells, rename/delete variables, filter rows, undo/redo (15-step history), autosave crash recovery |
| **Data Visualization** | Histograms, scatterplots, bar charts, line plots, moderation/interaction charts with 95% CI fit lines |
| **Descriptives & Crosstabs** | Means, SDs, skewness/kurtosis, Shapiro-Wilk normality, frequency tables, Chi-square contingency tables |
| **Item Analysis & CVI** | Scale reliability (Cronbach's α, McDonald's ω), corrected item-total correlations, Content Validity Index |
| **Correlation** | Pearson, Spearman, point-biserial, partial correlation; APA-formatted correlation matrix |
| **Compare Means** | Independent-samples *t*-test, paired *t*-test, one-sample *t*-test, Mann-Whitney U, Wilcoxon signed-rank |
| **Analysis of Variance** | One-way ANOVA, two-way factorial ANOVA, repeated-measures ANOVA, MANOVA, ANCOVA; Tukey HSD post-hoc |
| **Regression** | OLS, hierarchical (block-entry), binary/ordinal logistic, Ridge, Lasso; VIF, Breusch-Pagan, Durbin-Watson diagnostics |
| **Mediation Analysis** | Baron-Kenny steps + bootstrapped indirect effects, Sobel test, path diagrams |
| **Categorical PCA (CATPCA)** | PCA for mixed/ordinal data via sklearn; scree plots, biplot |
| **Cluster & Profile Analysis (LCA)** | K-Means, Agglomerative, Gaussian Mixture models; silhouette, dendrogram, profile plots |
| **Forecasting (Time Series)** | Holt-Winters exponential smoothing; trend/seasonal decomposition; forecast plots |
| **Exploratory EFA** | Factor analysis via factor_analyzer; parallel analysis, KMO, Bartlett; varimax/oblimin rotation |
| **Confirmatory CFA** | CFA via semopy; fit indices (CFI, TLI, RMSEA, SRMR, AIC, BIC); modification indices |
| **SEM (Graph & Syntax)** | Full structural equation modeling via semopy with lavaan-style syntax; standardized residuals |
| **Network Analysis** | Graph construction, centrality metrics (degree, betweenness, closeness, eigenvector), community detection |
| **Power Analysis** | A priori sample-size planning and sensitivity analysis for *t*-tests, ANOVA, correlation, Chi-square, proportions |

Additional capabilities:
- **Light & Dark mode** toggle
- **APA 7 formatted output** throughout (tables, write-up paragraphs)
- **Zoomable matplotlib charts** with the standard navigation toolbar
- **Variable labels, value labels, and measurement-scale metadata**
- **Export** results to HTML; charts downloadable via toolbar
- **Autosave** every 2 minutes; crash-recovery on next launch

---

## Example Datasets

To help you get started, this repository includes two sample datasets:

- experiment_example.csv
- survey_example.csv

---

## Screenshots

> *(Add screenshots to a `/docs/screenshots/` folder and link them here)*

---

## Installation

### Option A — Download a pre-built installer (recommended)

| Platform | Download |
|----------|----------|
| Windows (64-bit) | [PsyStat-Setup-Windows.exe](https://github.com/terysetn02-source/psystat/releases/latest) |
| macOS (Universal) | [PsyStat-macOS.dmg](https://github.com/terysetn02-source/psystat/releases/latest) |

No Python required. Just install and run.

### Option B — Run from source

**Requirements:** Python 3.10 or later

```bash
git clone https://github.com/your-username/psystat.git
cd psystat
pip install -r requirements.txt
python psystat.py
```

---

## Dependencies

All dependencies are listed in `requirements.txt`. Key libraries:

- [PyQt6](https://pypi.org/project/PyQt6/) — GUI framework
- [pandas](https://pandas.pydata.org/) — data handling
- [numpy](https://numpy.org/) / [scipy](https://scipy.org/) — numerical computation
- [statsmodels](https://www.statsmodels.org/) — regression, ANOVA, time series
- [scikit-learn](https://scikit-learn.org/) — clustering, PCA, regularized regression
- [matplotlib](https://matplotlib.org/) — charts
- [factor_analyzer](https://github.com/EducationalTestingService/factor_analyzer) — EFA
- [semopy](https://semopy.com/) — CFA and SEM
- [networkx](https://networkx.org/) — network analysis

---

## Building Installers

See [`BUILD.md`](BUILD.md) for step-by-step instructions on producing the Windows `.exe` installer and the macOS `.dmg` using PyInstaller.

---

## Documentation

A full user manual is available in [`MANUAL.html`](MANUAL.html). Open it in any browser.

---

## License & Citation

Created by Tery Setiawan (Universitas Kristen Maranatha & Radboud University). Created for academic and research purposes. Please cite the creator if you use this tool in published empirical research.
PsyStat is released under the [MIT License](LICENSE).

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss significant changes.

---

## Acknowledgements

PsyStat is built on the shoulders of the scientific Python ecosystem. The APA formatting conventions follow the *Publication Manual of the American Psychological Association* (7th ed.).
