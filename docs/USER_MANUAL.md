# PsyStat User Manual
### Statistical Analysis Application for Psychology Researchers
*Version 1.1*

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Data Management](#2-data-management)
3. [Data Visualization](#3-data-visualization)
4. [Descriptives & Crosstabs](#4-descriptives--crosstabs)
5. [Item Analysis & CVI](#5-item-analysis--cvi)
6. [Correlation](#6-correlation)
7. [Compare Means](#7-compare-means)
8. [Analysis of Variance](#8-analysis-of-variance)
9. [Regression](#9-regression)
10. [Mediation Analysis](#10-mediation-analysis)
11. [Categorical PCA (CATPCA)](#11-categorical-pca-catpca)
12. [Cluster & Profile Analysis (LCA)](#12-cluster--profile-analysis-lca)
13. [Forecasting (Time Series)](#13-forecasting-time-series)
14. [Exploratory Factor Analysis (EFA)](#14-exploratory-factor-analysis-efa)
15. [Confirmatory Factor Analysis (CFA)](#15-confirmatory-factor-analysis-cfa)
16. [SEM (Graph & Syntax)](#16-sem-graph--syntax)
17. [Network Analysis](#17-network-analysis)
18. [Power Analysis](#18-power-analysis)
19. [SPSS Parity Reference](#19-spss-parity-reference)
20. [Frequently Asked Questions](#20-frequently-asked-questions)

---

## 1. Getting Started

PsyStat is a statistical application for psychology researchers. It comes as a fully self-contained installer — no Python installation, no terminal commands, and no additional dependencies required.

### 1.1 Download

PsyStat is available from two locations:

- **Website** — visit the PsyStat website and click **Download** to get the latest stable release.
- **GitHub Releases** — go to the PsyStat GitHub repository and click the **Releases** tab. Download the file for your operating system from the latest release.

### 1.2 Installation

Because PsyStat is a new application and not yet recognised by Microsoft or Apple's security systems, both Windows and macOS will display a warning the first time you run it. This is normal for newly released software and does not mean the app is harmful. Follow the steps below to get past these prompts.

**Windows**

1. Download the `.exe` installer file.
2. Double-click the installer. Windows SmartScreen may show a blue warning screen saying **"Windows protected your PC"**.
3. Click **More info** — a second line of text and a new button will appear.
4. Click **Run anyway**.
5. Follow the on-screen installation steps.
6. Once installed, launch PsyStat from the Start Menu or the desktop shortcut.

> If you see a User Account Control (UAC) prompt asking *"Do you want to allow this app to make changes to your device?"*, click **Yes** to continue.

**macOS**

1. Download the `.dmg` disk image.
2. Open the `.dmg` and drag PsyStat into your **Applications** folder.
3. The first time you open PsyStat, macOS Gatekeeper will block it and show a message saying the app **"cannot be opened because it is from an unidentified developer"** or **"Apple cannot check it for malicious software"**. Click **OK** or **Done** to dismiss this dialog — do not try to open it again from the same double-click.
4. Open **System Settings** (macOS Ventura and later) or **System Preferences** (macOS Monterey and earlier).
5. Go to **Privacy & Security**.
6. Scroll down to the **Security** section. You will see a message saying PsyStat was blocked, with an **Open Anyway** button next to it.
7. Click **Open Anyway**. macOS will ask you to confirm — click **Open** in the dialog that appears.
8. PsyStat will now launch. From this point on, you can open it normally from your Applications folder or Launchpad without any further warnings.

> If the **Open Anyway** button does not appear in Privacy & Security, try right-clicking the PsyStat icon in your Applications folder and selecting **Open** from the menu. A dialog will appear with an **Open** button that bypasses Gatekeeper.

### 1.3 Navigating the App

PsyStat opens with a top navigation bar showing all 17 analysis modules arranged across two rows. Click any module button to switch to it. The left panel contains configuration controls; results appear in tabs on the right.

---

## 2. Data Management

Click **Data Management** in the navigation bar.

Use this module to load your dataset and prepare variables before running analyses. All transformation tools are organised into sub-tabs on the left panel.

### 2.1 Loading and Saving Data

- Click **📂 Load CSV/Excel/SPSS** to open a file. PsyStat accepts `.csv`, `.xlsx`, and `.sav` files.
- Click **💾 Save Dataset** to save the current state of your dataset (including any recodes or computed variables) back to a file.
- The right panel shows a **Dataset** tab (spreadsheet view of your data) and a **Variable** tab (variable names and types). You can rename variables directly in the Variable tab. Click **✚ Add Row** or **✚ Add Variable** to manually add data.

### 2.2 Compute & Transform Tab

Use this sub-tab to create new variables from existing ones.

**Compute Variables (Sum / Mean / Subtract / Z-Score)**

1. Under **Compute Variables**, select the variables you want to combine.
2. Type a name for the new variable in **New Variable Name**.
3. Click one of the four operation buttons:
   - **Sum (Total)** — adds selected variables together row by row
   - **Mean** — averages the selected variables row by row
   - **Subtract (V1 - V2)** — subtracts the second selected variable from the first
   - **Z-Score** — standardises selected variables (mean = 0, SD = 1)

**Duplicate & Convert Variable**

1. Select the original variable from the **Original Variable** dropdown.
2. Type a name in **New Name**.
3. Click **Duplicate (Force to Numeric)** — converts text categories to numeric codes.

**Create Interaction & Dummies**

- **Interaction (Var1 × Var2):** Select two variables from the dropdowns and click **Multiply & Create** to generate a product term for moderation or interaction analyses.
- **Create Dummy Variable:** Select a variable, choose a method (Median Split, Mean Split, or One-Hot Encoding), then click **Generate Dummy**.

### 2.3 Recode & Reverse Tab

**Recode Variable**

1. Select the variable to recode from the **Variable** dropdown.
2. Type your recoding rules in **Rules** using the format `OldValue=NewValue` (e.g., `Male=1, Female=2, 99=NaN`).
3. Optionally check **Create as new variable** and enter a name to keep the original intact.
4. Optionally check **Force to Numeric** to convert any unmatched text to missing.
5. Click **Recode Values**.

**Reverse Variables**

1. Select the items to reverse in the list.
2. Set the scale **Min** and **Max** values (e.g., 1 and 5 for a 5-point Likert scale).
3. Check **Create as new variable (append '_Rev')** if you want to preserve the originals.
4. Click **Reverse Selected**.

### 2.4 Remove Cases Tab

- **Remove by condition:** Select a variable, choose an operator (==, !=, >, >=, <, <=, contains, is missing), and type a value. Click **🗑 Remove Matching Rows**.
- **Remove by row number:** Type row numbers or ranges (e.g., `3, 7, 15-20`) in the field and click **🗑 Remove by Row Number**. Row numbers are 1-based.
- Use **Ctrl+Z** (Windows) or **Cmd+Z** (macOS) to undo a removal.

---

## 3. Data Visualization

Click **Data Visualization** in the navigation bar.

### 3.1 How to Run

1. Under **Data Selection**, choose a **Plot Type**:
   - **Histogram** — distribution of a single variable
   - **Scatterplot** — relationship between two variables with a smooth fit line
   - **Bar Chart** — frequency counts of a categorical variable
   - **Line Plot** — values of Y across values of X
   - **Moderation Chart (Interaction)** — scatterplot split by levels of a moderator variable
2. Select the **X-Axis Variable**. For Scatterplot and Line Plot, also select a **Y-Axis (Target Variable)**.
3. For **Moderation Chart**, also select a **Moderator (Z) Variable** (enabled automatically when this plot type is chosen).
4. Under **Aesthetics & Design**, choose a **Color Palette** (Standard, Pastel, Seaborn, Monochrome), toggle **Show Gridlines**, and adjust **Transparency** with the slider.
5. Click **▶ Generate Graph**.
6. Click **↗ Pop Out Chart** to open the current chart in a larger resizable window with full zoom, pan, and save controls.

---

## 4. Descriptives & Crosstabs

Click **Descriptives & Crosstabs** in the navigation bar.

### 4.1 Descriptive Statistics

1. Under **Descriptive Statistics**, select the variables you want to summarise.
2. Check **Generate Descriptives (Mean, SD, etc.)** to get means, standard deviations, min, max, skewness, and kurtosis.
3. Check **Generate Frequencies** to get a frequency count table for each selected variable.
4. Click **▶ Run Selected Analysis**.

Skewness values above |2| and kurtosis above |7| are flagged in red as potentially problematic for parametric tests (Kim, 2013).

### 4.2 Crosstabs (Chi-Square)

1. Under **Crosstabs (Contingency Table)**, select one or more variables for **Row Variables** and one or more for **Column Variables**.
2. Check any combination of: **Row Percentages**, **Column Percentages**, **Total Percentages**, and **Chi-Square Test** (checked by default).
3. Click **▶ Run Crosstab**.

---

## 5. Item Analysis & CVI

Click **Item Analysis & CVI** in the navigation bar.

### 5.1 Classical Item Analysis

Use this to evaluate the psychometric quality of individual items in a scale.

1. Under **Classical Item Analysis**, select all the items belonging to the scale.
2. Optionally, type a **Scoring Key** — the correct answer (e.g., `1` or `A`). If provided, responses matching the key are recoded to 1 (Correct) and all others to 0 (Incorrect). Leave blank to use raw scores.
3. Click **▶ Run Item Analysis**.

The output shows overall Cronbach's α and McDonald's ω for the scale, plus a per-item table with Item Difficulty (Mean), Item Discrimination (Item-Rest Correlation), Cronbach's α if Deleted, and Factor Loading (λ).

### 5.2 Content Validity Index (I-CVI)

Use this to calculate item-level and scale-level content validity from expert ratings.

1. Under **Content Validity Index (I-CVI) Calculator**, select the columns containing expert ratings (one column per expert rater).
2. Set the **Threshold** — the minimum score above which a rating counts as "relevant" (default: 0.5, meaning ratings greater than 0.5 are considered relevant).
3. Click **▶ Calculate I-CVI**.

The output shows I-CVI per item, the Scale-level CVI (S-CVI/Ave), and the S-CVI Universal Agreement (S-CVI/UA) with guideline flags.

---

## 6. Correlation

Click **Correlation** in the navigation bar.

### 6.1 How to Run

1. Select a **Method**: **Pearson (Parametric)** for continuous normally distributed variables, or **Spearman (Non-Parametric)** for ordinal data or when normality is violated.
2. Choose **Missing Value Handling**:
   - **Listwise Deletion** — drops any row with missing values across all selected variables; every cell uses the same N.
   - **Pairwise Deletion** — each pair uses all available data; N may differ between cells.
3. Choose a **Multiple Comparisons Correction** if you are testing many pairs simultaneously:
   - **None** — raw p-values, no correction
   - **Bonferroni** — strict; controls the probability of any false positive
   - **Benjamini-Hochberg (FDR)** — less conservative; controls the proportion of false positives among significant results
4. Select all variables to include in the **Select Variables** list.
5. Click **▶ Run Correlation**.

The output shows a correlation matrix with r values, p-values, significance stars, and (for pairwise deletion) per-cell N. A lower-triangle heatmap is generated automatically.

---

## 7. Compare Means

Click **Compare Means** in the navigation bar.

### 7.1 How to Run

1. Under **Test Design**, select:
   - **Independent Samples** — two separate groups (e.g., treatment vs. control)
   - **Paired Samples** — the same subjects measured twice (e.g., pre vs. post)
2. Under **Assumption**, select:
   - **Parametric (T-Tests)** — for normally distributed continuous data
   - **Non-Parametric (U/W)** — Mann-Whitney U (independent) or Wilcoxon W (paired) for ordinal data or violated normality
3. Optionally check **Use Welch's correction if Levene's test is significant**. When checked, Levene's test for equality of variances is run automatically, and Welch's t-test is applied if variances are unequal. This is recommended.
4. Select the **Dependent Variable** (the continuous outcome) and **Grouping Variable** (the categorical variable defining the two groups).
5. Click **▶ Run Comparison**.

For independent samples, a raincloud plot (combining box plot, violin, and raw data points) is generated automatically alongside the results.

---

## 8. Analysis of Variance

Click **Analysis of Variance** in the navigation bar.

### 8.1 Between-Subjects (ANOVA / MANOVA)

Use this for comparing means across three or more independent groups, or testing the effect of categorical factors on one or more continuous outcomes.

1. Under **Design**, select **Between-Subjects (ANOVA / MANOVA)**.
2. Move your continuous outcome variable(s) into **Dependent Variables**. Select one for ANOVA; select more than one for MANOVA.
3. Move your categorical grouping variable(s) into **Fixed Factors (Categorical)**.
4. Optionally move continuous control variables into **Covariates (Optional Metric)** for ANCOVA.
5. Check **Full Factorial (Include Interaction Effects for Factors)** to test interaction effects between factors. When checked, PsyStat automatically uses Type III Sums of Squares with sum coding — this matches the SPSS GLM default and is required for correct main-effect tests in unbalanced interaction designs.
6. Optionally check **Include Descriptive Statistics (Mean, SD, N per group)**.
7. Optionally check **Also run Kruskal-Wallis** for a non-parametric alternative (single DV, single factor only).
8. Click **▶ Run Analysis of Variance**.

### 8.2 Repeated Measures (RM-ANOVA / Friedman)

Use this when the same subjects were measured at three or more time points or conditions.

1. Under **Design**, select **Repeated Measures (RM-ANOVA / Friedman)**.
2. In **Repeated Measures (select 3+, in time order)**, select the columns representing each time point or condition in order.
3. Under **Test**, choose **Repeated-Measures ANOVA (Parametric)**, **Friedman Test (Non-Parametric)**, or **Both**.
4. Optionally check **Include Descriptive Statistics (Mean, SD, N per timepoint)**.
5. Click **▶ Run Repeated Measures Test**.

**Sphericity** is tested automatically via Mauchly's test. The output shows a colour-coded banner:
- **Green ✔** — Sphericity holds; use the uncorrected p-value.
- **Red ⚠** — Sphericity violated; use the Greenhouse-Geisser corrected p-value shown in the banner. If ε (GG) ≥ .75, the Huynh-Feldt correction is a less conservative alternative.

---

## 9. Regression

Click **Regression** in the navigation bar.

### 9.1 Choosing a Regression Type

Select the model type from the **Regression Model Type** dropdown at the top:

| Option | Use when |
|---|---|
| **Linear Regression (OLS)** | Continuous outcome |
| **Logistic Regression (Binary Y)** | Binary outcome (two categories) |
| **Multinomial Logistic Regression (Nominal Y, 3+ Categories)** | Unordered multi-category outcome |

### 9.2 How to Run

1. Select a **Dependent Variable (Y)**.
2. Move predictors into **Block 1 (Controls / All Predictors for Non-OLS)**. For hierarchical OLS regression, use multiple blocks — Block 1 for controls, Block 2 for main predictors of interest.
3. For OLS only, optionally check:
   - **Stepwise** — automatically enters predictors based on significance
   - **Standardized Coefficients (β)** — reports standardised betas alongside B
4. Click **▶ Run Regression**.

### 9.3 Pseudo R² for Logistic Regression

Three pseudo-R² values are reported for logistic and multinomial models:

| Metric | Comparable to SPSS? |
|---|---|
| **McFadden's R²** | Not shown in SPSS by default; systematically lower |
| **Cox & Snell R²** | ✔ SPSS default |
| **Nagelkerke R²** | ✔ SPSS default; rescaled to 0–1 range |

Use Cox & Snell and Nagelkerke when comparing to SPSS output. The log-likelihoods for both null and fitted models are shown for manual verification.

---

## 10. Mediation Analysis

Click **Mediation Analysis** in the navigation bar.

### 10.1 How to Run

1. Select the **Independent Variable (X)**, **Mediator (M)**, and **Dependent Variable (Y)** from the three dropdowns. All three must be different variables.
2. Under **Indirect Effect Significance Test**, set:
   - **Bootstrap Resamples** — number of bootstrap samples for the confidence interval (default: 2000; range: 500–10,000). More resamples = more precision but slower.
   - **Confidence Level** — 90%, 95%, or 99% CI.
3. Click **▶ Run Mediation Analysis**.

### 10.2 Output

The results show the four path coefficients (a, b, c, c′) in a table, followed by a highlighted summary box showing the indirect effect (a × b) with its bootstrap confidence interval, and the Sobel test as a secondary check.

- If the bootstrap CI **excludes zero** → indirect effect is significant (mediation supported).
- If the bootstrap CI **includes zero** → indirect effect is not significant.

Two additional tabs are generated automatically: a **Path Diagram** and a **Bootstrap Distribution** histogram.

---

## 11. Categorical PCA (CATPCA)

Click **Categorical PCA (CATPCA)** in the navigation bar.

### 11.1 How to Run

1. In **Select Variables to Reduce**, select the variables to include (minimum 3).
2. Set **Number of Components to Extract** using the spinner (default: 2).
3. Under **Missing Value Handling**, choose:
   - **Drop missing rows (Listwise)** — removes rows with any missing value
   - **Impute missing as separate category** — treats missing as its own category level
4. Click **▶ Run CATPCA**.

### 11.2 Scaling Levels

PsyStat auto-detects scaling levels from column types. The **Variable Scaling Levels** table in the results confirms what was detected:

| Detection rule | Scaling assigned |
|---|---|
| Text or category dtype | Nominal |
| Integer with ≤ 10 unique values | Ordinal |
| Float or integer with > 10 unique values | Numeric |

Based on the detected levels, PsyStat automatically selects the algorithm:
- All Nominal/Ordinal → **Multiple Correspondence Analysis (MCA)**
- Any Numeric present → **Factor Analysis of Mixed Data (FAMD)**

### 11.3 Reading the Output

- **Variable Scaling Levels** — confirms the detected scaling level for each variable.
- **Model Summary** — eigenvalues, % variance explained, cumulative %, and **Cronbach's α per dimension**. Retain dimensions with eigenvalue ≥ 1 and positive alpha. Negative alpha means the dimension should not be retained.
- **Category Quantifications** — optimal numeric value assigned to each category level (equivalent to SPSS "Category Quantifications"). Values ≥ |.40| are bolded.
- **Biplot** — category arrows (quantifications) and individual object scores (grey dots) on Dimension 1 vs. Dimension 2.

---

## 12. Cluster & Profile Analysis (LCA)

Click **Cluster & Profile Analysis (LCA)** in the navigation bar.

### 12.1 How to Run

1. Under **Variables (Indicators)**, select the variables that define the profiles (minimum 2).
2. Set the **Extraction Method**:
   - **Latent Profile Analysis (Gaussian Mixture)** — probabilistic; provides AIC, BIC, and entropy; recommended for psychological profiling
   - **K-Means Clustering** — deterministic; best for well-separated, roughly equal-size clusters
   - **Hierarchical Clustering** — builds a tree of nested clusters; useful for exploratory work
3. Set **Number of Classes/Clusters (k)** — the number of distinct subgroups to extract.
4. Click **▶ Run Latent Profile / Cluster Analysis**.

### 12.2 Choosing the Right k

For Latent Profile Analysis, run the analysis multiple times with different values of k and compare the AIC and BIC values. Lower AIC/BIC indicates a better fit. An entropy value ≥ .80 indicates clean class separation.

### 12.3 Output

- **Model Fit** (LPA only) — AIC, BIC, entropy
- **Class Sizes** — count and percentage of cases assigned to each class
- **Class Profiles (Raw Means)** — average score on each indicator variable per class; use this to name and interpret the classes
- **Plot** — 2D PCA scatter plot with classes colour-coded

---

## 13. Forecasting (Time Series)

Click **Forecasting (Time Series)** in the navigation bar.

### 13.1 Choosing a Method

Select the method from the **Method** dropdown at the top:

- **Exponential Smoothing (Single Series)** — for a single time series; projects future values based on recent levels and trend
- **Latent Growth Curve Modeling (LGCM)** — for repeated-measures panel data; estimates intercept (baseline level) and slope (rate of change) as latent factors

### 13.2 Exponential Smoothing

1. Select the **Time Variable (X)** and **Target Variable to Forecast (Y)**.
2. Set **Forecast Steps** — how many future periods to predict.
3. Click **▶ Run Model**.

The output shows model fit (AIC, BIC), smoothing parameters (α for level, β for trend), and a forecast plot showing historical data and projected values.

### 13.3 Latent Growth Curve Modeling (LGCM)

1. Under **Select Repeated Measures (T1, T2, T3... in order)**, select the columns representing each time point in chronological order (minimum 3).
2. Click **▶ Run Model**.

The output shows model fit indices (CFI, TLI, RMSEA, SRMR) and parameter estimates for the intercept (i) and slope (s) factors. A **Growth Plot** tab shows observed means vs. the implied growth trajectory.

---

## 14. Exploratory Factor Analysis (EFA)

Click **Exploratory (EFA)** in the navigation bar.

### 14.1 How to Run

1. In **Select Items for EFA**, select all items to include (minimum 3).
2. Choose an **Extraction Method**:
   - **minres (MINRES / PAF)** — default; equivalent to SPSS's Principal Axis Factoring
   - **ml (Maximum Likelihood)** — provides χ² and RMSEA fit indices; requires multivariate normality
   - **principal (PCA)** — extracts components (not factors); does not model unique variance
3. Choose a **Rotation Method**:
   - **promax** — oblique; allows factors to correlate (recommended default for most psychological scales)
   - **oblimin** — oblique alternative to promax
   - **varimax** — orthogonal; forces factors to be uncorrelated
   - **none** — no rotation; use only when extracting a single factor
4. Set **Number of Factors** — enter a specific number, or set to **0** for automatic selection using the Kaiser criterion (eigenvalue > 1).
5. Set **Suppress Loadings <** — hides loadings below this threshold in the output table (default: 0.30).
6. Click **▶ Run EFA**.

### 14.2 Output Tabs

**Assumptions & Scree tab:**
- KMO sampling adequacy (should be ≥ .60)
- Bartlett's test (should be p < .05)
- Total variance explained table
- Scree plot (red dashed line marks eigenvalue = 1)

**Matrix & Corr tab:**
- Factor loadings matrix (loadings ≥ .40 bolded; loadings below suppression threshold hidden)
- Communalities (h²) and Uniqueness per item
- Internal consistency (Cronbach's α and McDonald's ω) per factor
- Factor correlation matrix (oblique rotations only)
- SPSS comparison note explaining expected minor numerical differences from PAF

---

## 15. Confirmatory Factor Analysis (CFA)

Click **Confirmatory (CFA)** in the navigation bar.

### 15.1 How to Run

1. In **Your Lavaan Syntax** editor, type your measurement model using lavaan syntax. Use `=~` to assign indicators to latent factors:

```
Anxiety    =~ item1 + item2 + item3
Depression =~ item4 + item5 + item6
```

   The example shown above the editor (read-only) demonstrates the correct format.

2. Click **▶ Run CFA**.

### 15.2 Output — CFA Results Tab

**Model Information** — estimator, optimisation method, number of parameters, and number of observations.

**Model Fit Indices** — χ², df, p-value, baseline χ², CFI, TLI, loglikelihood, AIC, BIC, RMSEA, and SRMR. Guidelines (Hu & Bentler, 1999): excellent fit requires CFI ≥ .95, TLI ≥ .95, RMSEA ≤ .06, SRMR ≤ .08.

**Parameter Estimates table** — one sub-table per parameter type (Latent Variables, Regressions, Covariances, Intercepts, Variances), each showing:

| Column | What it shows |
|---|---|
| **Estimate** | Unstandardized coefficient (raw metric) |
| **Std.Err** | Standard error of the unstandardized estimate |
| **z-value** | Wald z-statistic |
| **P(>|z|)** | p-value (significant values highlighted) |
| **Std.lv** | Standardized for latent variables only (latent SD = 1; observed metric preserved) |
| **Std.all** | Fully standardized (all SD = 1; comparable to a correlation) |

Standardized loadings ≥ .50 are bolded in the Latent Variables table. SE, z, and p apply to the unstandardized estimate only.

**Construct Reliability and Validity table** — for each latent factor: Cronbach's α (from raw item scores), McDonald's ω / Composite Reliability (from Std.all loadings), and Average Variance Extracted (AVE). Guidelines (Fornell & Larcker, 1981): AVE ≥ .50 and CR ≥ .70 indicate good construct validity.

### 15.3 Output — Modification Indices Tab

Shows standardized residuals and modification indices — the largest values indicate where adding paths or covariances would most improve model fit.

---

## 16. SEM (Graph & Syntax)

Click **SEM (Graph & Syntax)** in the navigation bar.

### 16.1 How to Run via Syntax

1. In the **Lavaan Syntax** editor, type your full structural model using lavaan syntax. Use `=~` for measurement paths, `~` for structural regressions, and `~~` for covariances.
2. Click **▶ Estimate Structural Model**.

### 16.2 How to Run via the Interactive Builder

Use the **SEM (Graph & Syntax)** tab's visual builder as follows:

1. **Add Latent Factors** — type a factor name (no spaces) in the **Add Latent Factor** field and click **✚ Add Latent**, or press Enter.
2. **Add Observed Variables** — select a column from the **Add Observed Variable** dropdown and click **✚ Add Observed**.
3. **Draw Paths** — click **➚ Draw Path**, then click the source node, then click the target node to draw an arrow between them.
4. **Generate Syntax** — click **Generate Syntax ↓** to convert your diagram into lavaan syntax, which is automatically transferred to the syntax editor.
5. Click **▶ Estimate Structural Model** to run.

### 16.3 Output

The output is the same report format as CFA (§15.2), including Model Fit Indices, Parameter Estimates with Unstandardized, Std.lv, and Std.all columns, and Modification Indices.

---

## 17. Network Analysis

Click **Network Analysis** in the navigation bar.

### 17.1 How to Run

1. In **Select Variables for Network**, select the variables (nodes) to include (minimum 2).
2. Set **Minimum Partial Correlation Threshold (Absolute |r|)** — only edges with an absolute partial correlation at or above this value are drawn (default: 0.15). Increasing this produces a sparser, cleaner network; decreasing it shows more connections.
3. Choose a **Graph Layout**:
   - **Spring Layout (Fruchterman-Reingold)** — positions highly connected nodes near the centre; default
   - **Circular Layout** — equally spaced around a circle
   - **Kamada-Kawai Layout** — minimises edge crossings; good for dense networks
4. Click **▶ Run Network Analysis**.

### 17.2 Output

**Network Metrics tab** — a table showing for each node (variable):
- **Node Strength** — sum of absolute partial correlation weights; indicates overall influence in the network
- **Degree Centrality** — proportion of possible connections that are active
- **Betweenness Centrality** — how often a variable sits on the shortest path between other variables (bridge nodes)
- **Closeness Centrality** — how quickly a variable can influence all others

**Network Plot tab** — an interactive graph where:
- **Node size** scales with Node Strength
- **Green solid edges** = positive partial correlations
- **Red dashed edges** = negative partial correlations
- **Edge thickness** scales with correlation strength
- **Nodes are draggable** — click and drag any node to rearrange the layout

The network is based on a Gaussian Graphical Model (GGM): edges represent partial correlations controlling for all other variables, not zero-order correlations.

---

## 18. Power Analysis

Click **Power Analysis** in the navigation bar.

### 18.1 Choosing a Mode

Select the mode from the **Mode** dropdown:

| Mode | When to use |
|---|---|
| **A Priori: Find Required Sample Size (N)** | Before data collection — calculates how many participants you need |
| **Post-Hoc: Find Achieved Power (given N)** | After data collection — calculates the power your study actually had |
| **Sensitivity: Find Minimum Detectable Effect (given N and Power)** | After data collection — finds the smallest effect your study could have detected |

### 18.2 Choosing a Test

Select the statistical test from the **Statistical Test** dropdown. Available tests and when to use them:

| Test | Use when |
|---|---|
| **Independent T-Test** | Comparing two separate groups on one continuous outcome |
| **Paired T-Test** | Comparing two related measurements from the same subjects |
| **ANOVA: One-Way** | Comparing 3+ independent groups on one continuous outcome |
| **ANOVA: Two-Way / Factorial** | Two categorical factors; power for a main effect or their interaction |
| **Multiple Regression (Omnibus R²)** | Overall explanatory power of a full regression model |
| **Multiple Regression (Interaction / R² Increase)** | Power for a specific block of predictors added to an existing model |
| **Logistic Regression (Single Continuous Predictor)** | Binary outcome with one continuous predictor; detects a given odds ratio |
| **Correlation (Pearson r)** | Testing whether a correlation is significantly different from zero |
| **Chi-Square Test** | Association between categorical variables on frequency counts |
| **Mediation (X→M→Y Indirect Effect)** | Guidance only — no closed-form power calculator available |

### 18.3 Inputs

Depending on your mode and test selection, enter:

- **Effect Size** — the expected or observed effect (Cohen's d for t-tests, f for ANOVA, f² for regression, r for correlation, w for chi-square, odds ratio for logistic regression). Cohen's conventions: Small = .10, Medium = .30, Large = .50 (for r); Small = .20, Medium = .50, Large = .80 (for d).
- **Alpha (α)** — significance level (default: .05)
- **Target Power (1-β)** — desired power (default: .80; commonly .90 for grant applications)
- **Sample Size You Have (N)** — visible for Post-Hoc and Sensitivity modes only
- **Groups/Predictors** — number of groups (ANOVA) or predictors (regression), where applicable
- **Factor A/B Levels** and **Effect to Power** — visible for Two-Way ANOVA only

Click **▶ Calculate Power** to run. Results show the calculated value (required N, achieved power, or minimum detectable effect) with an interpretation note.

---

## 19. SPSS Parity Reference

| PsyStat module | Nav bar label | Parity level | Notes |
|---|---|---|---|
| **Factorial ANOVA** | Analysis of Variance | ✅ High | Type III SS + sum coding applied automatically with interactions |
| **Repeated-Measures ANOVA** | Analysis of Variance | ✅ High | Mauchly, GG, and HF corrections shown automatically |
| **Logistic Regression** | Regression | ✅ High | Cox & Snell and Nagelkerke R² reported alongside McFadden |
| **Multinomial Regression** | Regression | ✅ High | Same R² reporting as logistic |
| **CATPCA** | Categorical PCA (CATPCA) | ✅ High | MCA/FAMD optimal scaling; per-variable scaling detection; Cronbach's α per dimension |
| **EFA** | Exploratory (EFA) | ✅ High (minor diffs) | minres ≈ PAF; < 0.01 differences at 4th decimal expected |
| **CFA** | Confirmatory (CFA) | ✅ High | Unstandardized, Std.lv, and Std.all estimates; CR and AVE |
| **Descriptives** | Descriptives & Crosstabs | ✅ Exact | — |
| **Correlation** | Correlation | ✅ Exact | — |
| **Independent t-test** | Compare Means | ✅ Exact | — |
| **Paired t-test** | Compare Means | ✅ Exact | — |
| **One-Way ANOVA** | Analysis of Variance | ✅ Exact | — |
| **Chi-Square** | Descriptives & Crosstabs | ✅ Exact | — |
| **Multiple Regression (OLS)** | Regression | ✅ Exact | — |
| **Mediation** | Mediation Analysis | ✅ High | Bootstrap CI + Sobel; matches Hayes PROCESS Model 4 |
| **LCA / LPA** | Cluster & Profile Analysis (LCA) | ✅ High | GMM equivalent to Mplus LPA; K-Means also available |
| **Network Analysis** | Network Analysis | ✅ High | GGM partial correlations; matches qgraph/JASP network output |

**Parity levels:**
- ✅ **Exact** — Output matches SPSS to the displayed decimal places.
- ✅ **High** — Methodologically equivalent; minor numerical differences (< 0.01) may appear. Suitable for publication.
- ⚠ **Partial** — Core results match; some supplementary statistics differ.
- ❌ **Low** — Different algorithm; results not directly comparable.

---

## 20. Frequently Asked Questions

**Q: My ANOVA p-values changed when I checked "Full Factorial". Is something wrong?**

No. Checking Full Factorial switches from Type II to Type III Sums of Squares and applies sum coding automatically. This is the correct approach for interaction models with unbalanced data and matches SPSS GLM. For main-effects-only designs, leave Full Factorial unchecked.

**Q: The Repeated Measures output shows a red sphericity warning. What should I do?**

Use the Greenhouse-Geisser corrected p-value shown in the red banner rather than the uncorrected p-value from the ANOVA table. If ε (GG) ≥ .75, the Huynh-Feldt correction (also shown) is a slightly less conservative option. State in your report which correction you used.

**Q: The Regression output shows three different R² values. Which one matches SPSS?**

Use **Cox & Snell R²** and **Nagelkerke R²** for comparison with SPSS — these are what SPSS displays by default for logistic regression. McFadden's R² is included because it appears in many publications, but its values are systematically lower.

**Q: Can Cronbach's Alpha in CATPCA be negative?**

Yes, and SPSS reports negative values in the same circumstances. A negative alpha means the eigenvalue for that dimension is smaller than the number of variables — the dimension should not be retained.

**Q: Why do my CFA Std.lv and Std.all values differ?**

They measure different things. Std.lv standardises only the latent variable variances (latent SD = 1; observed item metrics preserved), while Std.all standardises everything (all SDs = 1, like a correlation). For AVE and CR calculations, Std.all is used. For interpreting how strongly an item loads on its factor relative to the item's own scale, Std.lv is more informative.

**Q: My EFA loadings differ from SPSS at the 4th decimal place. Is this a bug?**

No. PsyStat's minres method uses the SLSQP numerical optimiser while SPSS uses its own PAF routines. Both converge to the same solution with small floating-point differences at the 4th decimal place. A note explaining this appears at the bottom of every EFA output.

**Q: CATPCA auto-detected one of my Likert variables as "Nominal". How do I fix it?**

Auto-detection assigns Nominal to text/category dtype columns. If your Likert column is stored as text (e.g., "Strongly Agree"), go to **Data Management → Recode & Reverse** and recode the text responses to numeric values (1, 2, 3…). The column will then be detected as Ordinal automatically on the next CATPCA run.

**Q: The Network Analysis module says "Missing networkx module". What do I do?**

This should not occur in the standard installed version of PsyStat since all dependencies are bundled. If you see this message, try reinstalling PsyStat using the latest installer from the website or GitHub Releases page.

---

*PsyStat is developed as a research tool for academic and applied psychology. Results should be verified against published statistical theory before use in peer-reviewed publications. For questions or to report issues, use the in-app feedback form.*
