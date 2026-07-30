# PsyStat

**A Comprehensive Statistical Analysis Tool for Psychology & Social Science Research**

PsyStat is a modern, graphical desktop application designed for psychologists, social scientists, and researchers who need a powerful yet intuitive tool for statistical analysis. It bridges the gap between complex programming environments (R, Python) and expensive commercial software.

> Created by **Tery Setiawan**  
> Affiliations: Universitas Kristen Maranatha & Radboud University

---

## ✨ Key Features

| Category | What PsyStat Can Do |
|---|---|
| **Data Management** | Load CSV / Excel / SPSS (.sav), recode, reverse score, compute aggregates, create interaction & dummy variables |
| **Visualization** | Histograms, scatterplots, bar charts, line plots, moderation charts, raincloud plots, correlation heatmaps |
| **Descriptives** | Means, SDs, skewness, kurtosis, normality tests, frequency tables, crosstabs & chi-square |
| **Psychometrics** | Classical Item Analysis, Content Validity Index (I-CVI), EFA, CFA with Std.lv & Std.all |
| **Group Differences** | Independent & Paired T-Tests (with Welch's correction), Mann-Whitney U, Wilcoxon W |
| **ANOVA** | Between-subjects ANOVA/ANCOVA/MANOVA (Type III SS), Repeated-Measures ANOVA with Mauchly's sphericity test (GG & HF corrections), Friedman |
| **Regression** | OLS Linear, Binary Logistic, Multinomial Logistic (Cox & Snell, Nagelkerke R²) |
| **Advanced Modeling** | Mediation Analysis (bootstrap CI + Sobel), SEM with interactive visual builder, Latent Growth Curve Modeling |
| **Dimension Reduction** | CATPCA (MCA/FAMD), EFA (minres/ML/PCA), CFA via semopy |
| **Clustering** | Latent Profile Analysis (GMM), K-Means, Hierarchical Clustering |
| **Network Analysis** | Psychometric Network Analysis (Gaussian Graphical Model) with interactive graph |
| **Power Analysis** | A Priori, Post-Hoc, and Sensitivity modes for 10 statistical designs |
| **Forecasting** | Exponential Smoothing, Latent Growth Curve Modeling |
| **Export** | APA-formatted HTML/Word output for every analysis |

---

## 🚀 Installation

PsyStat is distributed as a **fully self-contained application** — no Python, no terminal, no dependencies to install.

### Windows
1. Go to the [**Releases**](../../releases) page and download `PsyStat-Setup.exe`
2. Double-click the installer
3. If Windows SmartScreen shows a blue warning: click **More info → Run anyway**
4. Launch PsyStat from the Start Menu or desktop shortcut

### macOS
1. Go to the [**Releases**](../../releases) page and download `PsyStat.dmg`
2. Open the `.dmg` and drag PsyStat to your **Applications** folder
3. On first launch, macOS will block the app — click **OK** to dismiss
4. Open **System Settings → Privacy & Security** and click **Open Anyway**
5. From this point on, PsyStat opens normally

> **Why the security warning?** PsyStat is a new application and has not yet been registered with Microsoft or Apple's code-signing infrastructure. The warning is standard for new software and does not indicate any risk.

---

## 📁 Repository Structure

```
psystat/
├── README.md                  ← You are here
├── LICENSE                    ← MIT License
├── CITATION.cff               ← Citation metadata
├── CHANGELOG.md               ← Version history
├── CONTRIBUTING.md            ← How to contribute or report bugs
├── psystat.py                 ← Full application source code
├── docs/
│   └── USER_MANUAL.md         ← Complete user manual (all 17 modules)
└── examples/
    ├── experiment_example.csv ← Sample experimental dataset (N=300)
    ├── survey_example.csv     ← Sample survey dataset (N=250)
    └── EXAMPLES_GUIDE.md      ← Dataset descriptions & suggested analyses
```

---

## 📖 Documentation

- **[User Manual](docs/USER_MANUAL.md)** — step-by-step guide for all 17 analysis modules
- **[Example Datasets](examples/)** — two ready-to-use datasets with suggested analyses
- **[Changelog](CHANGELOG.md)** — what changed in each version

---

## 🧪 Example Datasets

Two sample datasets are included in the `examples/` folder to help you get started immediately.

**`experiment_example.csv`** (N = 300) — An experimental intergroup contact study with pre/post measures. Suitable for: Paired T-Tests, Repeated-Measures ANOVA, Regression, Mediation Analysis, SEM.

**`survey_example.csv`** (N = 250) — A cross-sectional survey with Likert-scale items across four constructs. Suitable for: Item Analysis, EFA, CFA, Correlation, CATPCA, Network Analysis.

See [`examples/EXAMPLES_GUIDE.md`](examples/EXAMPLES_GUIDE.md) for full variable descriptions and suggested analysis workflows.

---

## 📝 Citation

If you use PsyStat in your research, please cite it:

```
Setiawan, T. (2025). PsyStat: A comprehensive statistical analysis tool for
psychology and social science research [Computer software].
Universitas Kristen Maranatha & Radboud University.
https://github.com/terysetn02-source/psystat
```

Or use the `CITATION.cff` file for automatic citation in supported tools (e.g., Zenodo, GitHub's Cite this repository button).

---

## 🤝 Contributing

Bug reports, feature suggestions, and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting.

---

## 📄 License

PsyStat is released for academic and research purposes under the MIT License. See [LICENSE](LICENSE) for full terms.
