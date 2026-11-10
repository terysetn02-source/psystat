# Changelog

All notable changes to PsyStat are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-11-01

### Added
- **Network analysis:** EBICglasso (graphical lasso with extended-BIC selection), significance-tested partial correlations
  (Bonferroni, Holm, Benjamini-Hochberg), unregularised partial correlations and zero-order correlation networks;
  Pearson, Spearman and nonparanormal input; expected influence; bootstrap edge-weight accuracy and case-dropping
  stability (CS-coefficient); custom node colours for user-defined subscales; adjustable edge colours, label font and
  node size; scrollable plot (mouse wheel, Ctrl+wheel zoom), *Open in Window* and *Save 300 DPI*.
- **CFA:** robust **MLR** estimator (Huber-White sandwich standard errors, Satorra-Bentler scaled chi-square, scaled
  CFI/TLI/NFI/RMSEA); **raw-data metric** (unstandardised estimates in the instrument's own scale); model-implied
  Std.lv and Std.all for all models.
- **Mediation:** R-squared, adjusted R-squared, F and delta-R-squared for each regression equation.
- **Bayesian analysis:** 95% credible intervals (CrI), BF01 next to BF10, prior specification in every table, posterior
  median and mean (exact posteriors for the t-tests and the correlation), frequentist statistics in a separate
  reference table, stretched-beta prior width setting for correlations, Bayesian bootstrap for mediation.
- **Documentation:** full user manual (Markdown, HTML, PDF), four simulated example datasets with an examples guide,
  README, contribution and security policies, citation file.
- **Packaging:** PyInstaller specification for PyQt6, Windows installer script, macOS disk-image script, GitHub Actions
  workflows (CI and release builds for Windows, macOS Apple Silicon and macOS Intel), and a `--selftest` command
  that verifies a built application.
- Autosave failures now show a warning dialog (once per failure episode).

### Changed
- CFA and SEM: **listwise deletion is now the default** missing-data method; FIML uses all rows but does not provide
  global fit indices (see Fixed).
- The network plot layout is defined in inches, so the title, plot and legend keep proportional margins at any size.
- Network centralities (closeness, betweenness) use weighted path lengths (1/|weight|), as in qgraph.
- *p*-values are shown without a leading zero (APA 7), e.g. *p* = .032.
- CVI input label clarified (use 2.5 for a 4-point scale).

### Fixed
- Kamada-Kawai layout crashed with negative edge weights; layouts now use positive strengths.
- Partial-correlation networks with a total score and its items returned arbitrary (sign-flipping) values; the singular
  case is now detected and explained.
- Bridge analysis used signed weights for community detection and betweenness; it now uses absolute weights and the same
  grouping as the plot.
- CFA/SEM reported "FIML" while silently dropping incomplete rows; FIML now really uses incomplete rows, and the invalid
  fit indices that follow a FIML fit are no longer shown.
- Factor loadings appeared under "Regressions" instead of "Latent Variables"; CFI could exceed 1; the baseline chi-square
  *p*-value silently failed.
- Optimizer stopped early on raw-scale data (inflated chi-square); a tighter tolerance is now used.
- Mediation crashed if it was the first analysis run in a session; its APA write-up always said "significantly predicted".
- Bayesian: the Cauchy prior scale *r* was ignored, the Welch *t* was used for the Bayes factor, and the correlation note
  named the wrong prior.
- Repeated-measures ANOVA silently skipped Mauchly's test unless pingouin had already been loaded, and read column names
  that newer pingouin versions renamed.
- Categorical PCA failed for all-numeric selections and for text columns under pandas 3; text-column detection no longer
  depends on the `object` dtype.
- Typing text into a numeric spreadsheet cell no longer corrupts the column; invalid entries are rejected with a message.
- Autosave folder could be unusable for a second user on shared Linux/macOS computers.
- Latent growth models no longer print misleading global fit indices.
- Menu labels in the Network module were clipped in narrow side panels.
- README/Guide menu entries opened bare Markdown files; they are now rendered to HTML. Previous build specification
  referenced PyQt5 although the application uses PyQt6.

## [1.1.0]

Earlier releases: see the [GitHub releases page](https://github.com/terysetn02-source/psystat/releases).
