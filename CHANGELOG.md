# Changelog

All notable changes to PsyStat are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2025-07-30

### Fixed — Statistical Corrections

- **Factorial ANOVA:** Switched from Type II to Type III Sums of Squares when
  interaction terms are included. Sum (effects) coding is now applied
  automatically to all categorical predictors in interaction models, matching
  SPSS GLM and R `car::Anova(type=3)` behaviour. Type II is still used for
  main-effects-only models.

- **Repeated-Measures ANOVA:** Replaced `statsmodels.stats.anova.AnovaRM`
  with `pingouin.rm_anova`. The output now includes Mauchly's test of
  sphericity (W and p), Greenhouse-Geisser ε and corrected p-value, and
  Huynh-Feldt ε and corrected p-value — matching SPSS GLM Repeated Measures
  output. A colour-coded banner (green ✔ / red ⚠) guides users on which
  p-value to report.

- **Logistic & Multinomial Regression:** Added Cox & Snell R² and Nagelkerke
  R² to the Model Fit table. These are the two metrics SPSS reports by
  default. McFadden's R² is retained but now explicitly labelled. Log-
  likelihoods for null and fitted models are shown for manual verification.

- **CATPCA:** Replaced `OrdinalEncoder + StandardScaler + PCA` pipeline with
  `prince.MCA` (for purely categorical data) and `prince.FAMD` (for mixed
  data). This implements optimal scaling equivalent to SPSS's Alternating
  Least Squares algorithm. Added per-variable scaling level detection
  (Nominal / Ordinal / Numeric) with auto-detection from column dtype. Added
  Cronbach's Alpha per dimension to the Model Summary table, matching SPSS
  CATPCA output.

- **CFA:** Fixed `Std.lv` and `Std.all` columns showing identical values.
  Both are now computed correctly via `model.inspect(std_est=...)` with a
  manual matrix-algebra fallback. Fixed Cronbach's α, McDonald's ω (CR), and
  AVE disappearing after the standardization fix by giving the reliability
  block its own independent standardized loading lookup.

- **EFA:** Added SPSS comparison disclaimer to every output explaining that
  minor differences from SPSS PAF at the 4th decimal place are expected and
  not errors.

---

## [1.0.0] — 2025-07-01

### Initial Release

- 17 analysis modules: Data Management, Data Visualization, Descriptives &
  Crosstabs, Item Analysis & CVI, Correlation, Compare Means, Analysis of
  Variance, Regression, Mediation Analysis, Categorical PCA, Cluster &
  Profile Analysis, Forecasting, Exploratory Factor Analysis, Confirmatory
  Factor Analysis, SEM, Network Analysis, Power Analysis.
- APA-formatted HTML output for every module.
- Export to HTML and Word.
- Built-in example datasets (`experiment_example.csv`, `survey_example.csv`).
- Self-contained installers for Windows (.exe) and macOS (.dmg).
