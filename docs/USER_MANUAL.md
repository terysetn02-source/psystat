# PsyStat User Manual

**Version 1.2.0** &nbsp;|&nbsp; A desktop statistics application for psychology and the social sciences

**Author:** Tery Setiawan (Universitas Kristen Maranatha & Radboud University)  
**Licence:** MIT &nbsp;|&nbsp; **Source code:** <https://github.com/terysetn02-source/psystat>

> **How to use this manual.** If you are new to PsyStat, read Chapters 1–4 (about 15 minutes) and then jump to the chapter for the analysis you need. Every analysis chapter has the same layout: *when to use it*, *how to run it*, *what you get*, and *how to report it*. Chapter 25 contains a step-by-step tutorial for every module using the example datasets that come with the app.

---

## Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick start (10 minutes)](#3-quick-start-10-minutes)
4. [The PsyStat interface](#4-the-psystat-interface)
5. [Data Management](#5-data-management)
6. [Data Visualization](#6-data-visualization)
7. [Descriptives & Crosstabs](#7-descriptives--crosstabs)
8. [Item Analysis & CVI](#8-item-analysis--cvi)
9. [Correlation](#9-correlation)
10. [Compare Means (t-tests)](#10-compare-means-t-tests)
11. [Analysis of Variance](#11-analysis-of-variance)
12. [Regression](#12-regression)
13. [Mediation Analysis](#13-mediation-analysis)
14. [Categorical PCA (CATPCA)](#14-categorical-pca-catpca)
15. [Cluster & Profile Analysis (LCA)](#15-cluster--profile-analysis-lca)
16. [Forecasting (Time Series) & Latent Growth Curves](#16-forecasting-time-series--latent-growth-curves)
17. [Exploratory Factor Analysis (EFA)](#17-exploratory-factor-analysis-efa)
18. [Confirmatory Factor Analysis (CFA)](#18-confirmatory-factor-analysis-cfa)
19. [Structural Equation Modeling (SEM)](#19-structural-equation-modeling-sem)
20. [Network Analysis](#20-network-analysis)
21. [Power Analysis](#21-power-analysis)
22. [Bayesian Analysis](#22-bayesian-analysis)
23. [Exporting, saving and settings](#23-exporting-saving-and-settings)
24. [Autosave and crash recovery](#24-autosave-and-crash-recovery)
25. [Tutorials with the example datasets](#25-tutorials-with-the-example-datasets)
26. [Troubleshooting & FAQ](#26-troubleshooting--faq)
27. [Statistical notes, limitations and references](#27-statistical-notes-limitations-and-references)
28. [How to cite PsyStat](#28-how-to-cite-psystat)

---

## 1. Introduction

### 1.1 What is PsyStat?

PsyStat is a free, open-source desktop application for statistical analysis, designed for students, psychologists and social scientists. It sits between two extremes: programming environments such as R or Python (powerful, but with a steep learning curve) and commercial point-and-click packages (easy, but expensive and rigid). With PsyStat you load a dataset, choose variables from a list, press **Run**, and receive APA-styled tables and publication-ready charts. Every analysis is computed with established scientific libraries (SciPy, statsmodels, pingouin, semopy, factor_analyzer, networkx and others), and many analyses also print **reproducible R or Python syntax** so you can verify or extend them.

### 1.2 What can it do?

| Area | Modules |
|---|---|
| **Data** | Import CSV / Excel / SPSS; spreadsheet-style editing; compute, recode, reverse-score, dummy-code; undo/redo; autosave |
| **Describe** | Data Visualization, Descriptives & Crosstabs, Item Analysis & CVI, Correlation |
| **Compare & predict** | Compare Means, ANOVA / ANCOVA / MANOVA / repeated measures, Regression (OLS with hierarchical blocks, polynomial, logistic, multinomial, ordinal, ridge / lasso), Mediation |
| **Dimension reduction & groups** | CATPCA, Cluster & Profile Analysis (LCA), Exploratory Factor Analysis |
| **Latent-variable models** | Confirmatory Factor Analysis, Structural Equation Modeling (graphical builder + syntax) |
| **Time & networks** | Forecasting (exponential smoothing), Latent Growth Curve Models, Psychometric Network Analysis |
| **Planning & inference** | Power Analysis (a priori, post-hoc, sensitivity), Bayesian Analysis (BF₁₀/BF₀₁, credible intervals) |

### 1.3 What PsyStat is *not*

* It is not a replacement for statistical training. The app explains each analysis and flags assumption problems, but **you** remain responsible for choosing appropriate methods and interpreting them.
* It does not currently support multilevel/mixed-effects models, polychoric/ordinal SEM estimators (WLSMV), multiple-group models, or imputation. Where a better option exists elsewhere, the app says so (for example, a warning appears when ordinal Likert indicators are used in CFA/SEM).
* Results are computed with documented, tested algorithms, but no software is free of bugs. For high-stakes work, verify key results with a second program, and please report discrepancies on GitHub.

### 1.4 System requirements

| | Windows | macOS |
|---|---|---|
| Operating system | Windows 10 or 11 (64-bit) | macOS 12 Monterey or newer |
| Processor | Any 64-bit CPU | Apple Silicon (M1–M4) **or** Intel |
| Memory | 4 GB (8 GB recommended) | 4 GB (8 GB recommended) |
| Disk space | about 1 GB | about 1 GB |
| Screen | 1366 × 768 or larger (1920 × 1080 recommended) | same |

---

## 2. Installation

### 2.1 Windows

1. Go to the **Releases** page of the project on GitHub and download `PsyStat-Setup-x.y.z.exe` (installer) — or `PsyStat-Windows-x.y.z.zip` if you prefer a portable copy that needs no installation.
2. Double-click the installer and follow the steps. PsyStat is added to the Start menu (and optionally the desktop).
3. **First launch — SmartScreen.** Because the app is distributed free of charge and is not signed with a paid code-signing certificate, Windows may show *"Windows protected your PC"*. Click **More info → Run anyway**. This happens once.
4. Start **PsyStat** from the Start menu. The first start can take 10–30 seconds while the scientific libraries load; later starts are faster.

### 2.2 macOS

1. Download the disk image that matches your Mac:
   * `PsyStat-x.y.z-macOS-AppleSilicon.dmg` for Macs with an M-series chip, or
   * `PsyStat-x.y.z-macOS-Intel.dmg` for Intel Macs.
   (Not sure? Apple menu → **About This Mac**: "Chip: Apple M…" means Apple Silicon; "Processor: Intel…" means Intel.)
2. Open the `.dmg` and drag **PsyStat** into the **Applications** folder.
3. **First launch — Gatekeeper.** The app is not notarised by Apple, so macOS will say it "cannot be opened because the developer cannot be verified". To open it once:
   * In **Applications**, **right-click (or Control-click) PsyStat → Open → Open**; or
   * Open **System Settings → Privacy & Security**, scroll to the message about PsyStat and click **Open Anyway**.
   After this, PsyStat opens normally with a double-click.
4. If macOS says the app is "damaged", open **Terminal** and run: `xattr -dr com.apple.quarantine /Applications/PsyStat.app`

### 2.3 Running from source (any system with Python)

Advanced users and contributors can run PsyStat directly from the source code. Python **3.11 or 3.12** is recommended.

```bash
git clone https://github.com/terysetn02-source/psystat.git
cd psystat
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python psystat.py
```

### 2.4 Updating and uninstalling

* **Update:** download the newest installer or disk image and install it over the old version. Your data files are never stored inside the app, so nothing is lost.
* **Uninstall (Windows):** Settings → Apps → PsyStat → Uninstall. **(macOS):** drag PsyStat from Applications to the Bin.

---

## 3. Quick start (10 minutes)

This walk-through loads sample data, runs three analyses and exports the results.

**Step 1 – Load the sample data.** In the menu bar choose **File → Load Sample Data → Experiment Example (N=300) — Contact Study**. The data appear in the **Dataset** tab of *Data Management*.

**Step 2 – Look at the data.** Scroll the table. Open the **Variable Info** tab to see each variable's type and measurement scale. Missing values appear as empty cells.

**Step 3 – Descriptives.** Click **Descriptives & Crosstabs** in the top navigation bar. In *Variables to Analyze* select `prejudice_post` and `contact_quality_post` (Ctrl/Cmd-click selects several), tick **Generate Descriptives**, and press **▶ Run Selected Analysis**. A table with N, mean, SD, skewness and kurtosis appears on the right.

**Step 4 – An independent-samples t-test.** Open **Compare Means**. Choose *Independent Samples*, set **Dependent Variable** = `contact_quality_post` and **Grouping Variable** = `condition`, and press **▶ Run Comparison**. You should see that the Contact group scored much higher than the Control group (t ≈ 10.9, large effect, Cohen's *d* ≈ 1.3). A raincloud plot is drawn on a second tab.

**Step 5 – A chart.** Open **Data Visualization**, choose **Plot Type = Bar Chart**, X = `condition`, Y = `prejudice_post`, and press **▶ Generate Graph**. Use **↗ Pop Out Chart** to open it in its own window.

**Step 6 – Export.** On any results panel click **💾 Export to HTML/Word**, choose a file name and open the file in a browser or in Word.

That is the whole workflow: **load → choose variables → run → read → export**. The remaining chapters explain each module.

---

## 4. The PsyStat interface

### 4.1 Overview

The window has four areas:

1. **Menu bar** (top): File, Edit, Settings, Help.
2. **Module bar** (below the menu): two rows of buttons, one per module. Click a button to switch module.
3. **Settings panel** (left of each module): variable lists, options and the **▶ Run** button. You can drag the divider between the left panel and the results to resize it, or drag it all the way left to hide the panel.
4. **Results panel** (right): every run opens a **new tab** stamped with the time (for example *T-Test (14:32:07)*), so earlier results are never overwritten. Close a tab with its ✕. Charts open as tabs too and can be popped out.

### 4.2 Menus

| Menu | Items |
|---|---|
| **File** | *New Window (Work on 2nd Dataset)*; *Open Dataset (CSV, Excel, SPSS)*; **Load Sample Data** (the four example datasets and the Examples Guide); *Save Dataset As…* |
| **Edit** | *Undo* (Ctrl/Cmd + Z), *Redo* (Ctrl/Cmd + Y) |
| **Settings** | *Toggle Light/Dark Mode*; *Decimal Formatting* (2, 3 or 4 decimals in all tables) |
| **Help** | *User Manual* (opens in the browser as HTML or as PDF); *README / About PsyStat*; *Example Datasets Guide*; *About* |

### 4.3 Choosing variables

Most modules use list boxes labelled for their purpose (for example *Dependent Variables*, *Fixed Factors*, *Covariates*). You can

* **click** a variable to select it, **Ctrl/Cmd-click** to add more, **Shift-click** to select a range;
* use the **🔍 Pop-up Select** button above a list to open a larger selection window (helpful for datasets with many variables), and **Clear** to empty the list;
* **drag** a variable name from the *Available Variables* panel onto a field (where the module offers it).

### 4.4 Reading results

Tables follow **APA 7 style**: *M*, *SD*, *t*, *F*, *p* are italicised, *p* values are reported without a leading zero (*p* = .032; *p* < .001), and significant results are highlighted. Under most tables a grey **Note** explains the statistic, states assumptions, and gives the reference. Many analyses end with a ready-to-edit **APA results sentence** and, where applicable, **Reproducible Syntax** in R and Python.

### 4.5 Light and dark mode

**Settings → Toggle Light/Dark Mode** switches the whole interface. Charts redraw in the matching colours.

### 4.6 Several datasets at once

**File → New Window (Work on 2nd Dataset)** opens a second, independent PsyStat window, for example to compare two data files side by side.

---

## 5. Data Management

*Module bar: **Data Management**.* This is where you load, inspect, edit and transform data. It has two tabs: **📊 Dataset** (the spreadsheet) and **📋 Variable Info** (a description of each variable).

### 5.1 Loading data

* **File → Open Dataset (CSV, Excel, SPSS)**, or the **📂 Load CSV/Excel/SPSS** button in Data Management.
* Supported formats: **.csv**, **.xlsx / .xls** (Excel) and **.sav** (SPSS).
* Put **variable names in the first row** and one participant per row. Avoid spaces and special characters in variable names where you can (`life_satisfaction_post` rather than `Life satisfaction (post)`), and code missing values as empty cells.
* To practise, use **File → Load Sample Data** (see Chapter 25).

### 5.2 The Dataset tab (spreadsheet)

Double-click a cell to edit it. Toolbar buttons:

| Button | What it does |
|---|---|
| 🆕 New Sheet | Start an empty dataset |
| ✚ Add Row / ✚ Add N Rows | Append one or *N* blank rows |
| ✚ Add Variable | Add a new empty variable (column) |
| 🗑 Delete Row(s) / 🗑 Delete Col | Delete the selected rows or column |
| ✏ Rename Column | Rename the selected variable |
| ⬇ Fill Down | Copy the value of the first selected cell down the selection |
| 📋 Paste from Clipboard | Paste a block copied from Excel or another table |
| 📄 Copy Selection | Copy the selected cells |
| 💾 Save Dataset | Save to CSV or Excel |
| ↩ Undo / ↪ Redo | Step backwards/forwards through your edits (also Ctrl/Cmd + Z / Y) |

**Data-entry safety.** When you type into a *numeric* column, PsyStat only accepts a valid number (a decimal comma such as `1,5` is understood as 1.5). If you type text such as `5.a2`, the entry is **rejected**, the previous value is restored and a message explains why. This prevents a single typo from turning a whole numeric column into text. A brand-new empty variable, or a column that already holds text, accepts text.

A **History** list shows the operations performed on the dataset in this session.

### 5.3 The Variable Info tab

One row per variable, with editable columns:

* **Variable Name**, **Variable Label** (a descriptive title) and **Value Labels** (for example `1=Low, 5=High`);
* **Measurement Scale**: *Nominal, Ordinal, Interval* or *Ratio*;
* **Data Type**: *Numeric (Float), Integer, String/Text* or *Category*;
* **Missing Values**: the number of empty cells (read-only).

Setting the correct scale and type helps PsyStat offer suitable analyses (for example, treating text as a grouping variable).

### 5.4 Transforming data

All tools below sit in the left panel of Data Management. Most have a **Create as new variable** checkbox: leave it ticked to keep the original variable and create a new one (recommended), or untick it to overwrite.

| Tool | Use it to | How |
|---|---|---|
| **Sum / Mean / Subtract / Z-Score** | Create a scale score or standardise | Select the variables under *Select Variables*, type a *New Name*, press **Sum (Total)**, **Mean**, **Subtract (V1−V2)** or **Z-Score** |
| **Duplicate & Convert Variable** | Make a numeric copy of a text variable | Pick the *Original Variable*, name the copy, tick **Force to Numeric (unmatched text becomes missing)** if needed, press **Duplicate** |
| **Interaction & Dummy Variables** | Build a product term for moderation, or dummy codes | **Multiply & Create** multiplies two variables. **Generate Dummy** offers *Median Split (0/1)*, *Mean Split (0/1)* and *One-Hot Encoding* |
| **Recode Variable** | Change values | Choose the variable and type rules such as `Male=1, Female=2, 99=NaN` (comma-separated; `NaN` means missing). Press **Recode Values** |
| **Reverse-Score Variables** | Reverse negatively worded Likert items | Select the items, enter the scale **Min** and **Max** (for a 1–5 item: 1 and 5), press **Reverse Selected**. A score *x* becomes *Min + Max − x* |
| **Remove Cases (Rows)** | Exclude participants | Choose a variable, a condition (`==`, `!=`, `>`, `>=`, `<`, `<=`, `contains`, `is missing`) and a value, then **Remove Matching Rows**; or list row numbers and press **Remove by Row Number** |

> **Tip.** Removing cases changes your data. Press **↩ Undo** if you remove the wrong rows, and save the dataset first (**💾 Save Dataset**) if you want a copy of the full data.

### 5.5 Saving

**File → Save Dataset As…** or **💾 Save Dataset** writes the current data to CSV or Excel. PsyStat also autosaves your work in the background (Chapter 24), but **always save your data yourself** before closing.

---

## 6. Data Visualization

*Module bar: **Data Visualization**.* Create publication-quality charts.

### 6.1 Running a chart

1. Choose a **Plot Type**: *Histogram, Scatterplot, Bar Chart, Line Plot* or *Moderation Chart (Interaction)*.
2. Pick the **X-Axis Variable**, the **Y-Axis (Target Variable)** and, for a moderation chart, the **Moderator (Z) Variable**.
3. Adjust **Aesthetics & Design**: *Color Palette* (Standard, Pastel, Seaborn, Monochrome), *Show Gridlines*, and *Transparency*.
4. Press **▶ Generate Graph**. Two tabs appear: **Graph** and **Chart Guide** (a plain-language explanation of how to read that chart).

Use **↗ Pop Out Chart** to open the chart in a separate window. Chart windows have a toolbar for zooming and panning, and a **Save 300 DPI** option for journal figures.

### 6.2 Which chart when?

| Question | Chart | Variables |
|---|---|---|
| What does one variable look like? | Histogram | X = numeric |
| How are two numeric variables related? | Scatterplot | X, Y numeric |
| Group means | Bar Chart | X = grouping (categorical), Y = numeric |
| Change over time | Line Plot | X = time, Y = numeric |
| Does the X→Y relationship depend on a third variable? | Moderation Chart | X, Y numeric; Z = moderator. Lines show the X→Y slope at low, average and high levels of Z |

---

## 7. Descriptives & Crosstabs

*Module bar: **Descriptives & Crosstabs**.*

### 7.1 Descriptive statistics and frequencies

1. Under **Variables to Analyze** select one or more variables (select none to analyse all).
2. Tick **Generate Descriptives (Mean, SD, etc.)** and/or **Generate Frequencies**.
3. Press **▶ Run Selected Analysis**.

**You get:** for each numeric variable *N (valid)*, *Mean* with its 95% CI, *Median*, *SD*, *SE*, *Min*, *Max*, *Skewness*, *Kurtosis*, and two normality tests (*Shapiro–Wilk* and *Kolmogorov–Smirnov with Lilliefors correction*). A table reports **missing values** per variable, and a note states how many rows were lost to listwise deletion. Frequency tables list counts and percentages for each category.

**How to read normality tests.** With large samples these tests flag trivial departures, and with small samples they lack power. Judge normality by the **skewness and kurtosis** (|skew| < 2 and |kurtosis| < 7 are common rules of thumb for approximate normality; West, Finch & Curran, 1995) *and* by a histogram, not by *p* alone.

### 7.2 Crosstabs

1. Select **Row Variables** and **Column Variables** (categorical).
2. Choose which percentages to show: **Row**, **Column**, **Total**; tick **Chi-Square Test** for a test of independence.
3. Press **▶ Run Crosstab**.

**You get:** the contingency table with counts and the chosen percentages, and the **Chi-Square Test of Independence** (χ², df, *p*). The note warns when expected cell counts are small (< 5), in which case the chi-square approximation is unreliable and categories should be combined or an exact test used.

---

## 8. Item Analysis & CVI

*Module bar: **Item Analysis & CVI**.* Two tools for building and evaluating questionnaires.

### 8.1 Classical item analysis (reliability)

**When to use it:** you have a multi-item scale and want its internal consistency and the quality of each item.

1. **Reverse-score negatively worded items first** (Chapter 5.4). Otherwise α is badly underestimated.
2. Under **Select Items to Analyze** choose the items of **one scale**.
3. Press **▶ Run Item Analysis**.

**You get:** the overall **Cronbach's α**, **McDonald's ω** (estimated from a one-factor model; it does not assume equal loadings and is often the more accurate estimate), and **split-half reliability** (Spearman–Brown); then, per item, *Mean (Difficulty)*, *SD*, *Item-Rest Correlation (Discrimination)*, *Cronbach's α if Deleted* and *Factor Loading (λ)*. A ready-made APA sentence summarises the reliability. A note reports how many rows were lost to listwise deletion.

**Interpreting.** α ≥ .70 is usually considered acceptable for research use (.80+ good), but α rises with the number of items and does not prove unidimensionality — check dimensionality with EFA/CFA. An item with an item-rest correlation below about .30, or whose removal *raises* α noticeably, is a candidate for revision.

### 8.2 Content Validity Index (I-CVI)

**When to use it:** experts have rated how relevant each item is, and you want agreement indices.

**Data layout.** One **row per item** and one **column per expert** (as in `cvi_expert_ratings_example.csv`).

1. Under **Select Expert Rating Columns** select all the expert columns.
2. Set **Rating counts as 'Relevant' if greater than**: for a 4-point relevance scale (1 = not relevant … 4 = highly relevant) enter **2.5** so that ratings of 3 and 4 count as relevant. For 0/1 coding enter **0.5**.
3. Press **▶ Calculate I-CVI**.

**You get:** for every item the number of experts rating it relevant and the **I-CVI** (relevant ratings ÷ number of experts), with a status, plus the scale-level **S-CVI/Ave** (mean I-CVI) and **S-CVI/UA** (proportion of items on which all experts agree).

**Guidelines (Lynn, 1986; Polit & Beck, 2006):** with 3–5 experts every item should reach I-CVI = 1.00; with 6 or more experts I-CVI ≥ .78 is acceptable. S-CVI/Ave ≥ .90 is recommended for the scale as a whole. Items marked *Revise/Reject* should be reworded or dropped.

---

## 9. Correlation

*Module bar: **Correlation**.*

1. Select **two or more variables**.
2. Choose the **Method**: *Pearson (Parametric)* for linear relationships between roughly normal continuous variables; *Spearman (Non-Parametric)* for ordinal data, outliers or monotonic non-linear relations.
3. Choose **Missing Value Handling**: *Listwise* (one common sample for all correlations) or *Pairwise* (uses all available data for each pair; the *N* then differs between cells).
4. Choose a **p-value correction** for the many tests in a matrix: *None (raw p-values)*, *Bonferroni*, or *Benjamini–Hochberg (FDR)*.
5. Press **▶ Run Correlation**.

**You get four tabs:** **Correlation** (the matrix), **p-values**, **95% CIs** (Fisher's *z* intervals) and a **Heatmap**.

**Interpreting.** Cohen's (1988) guidelines: |*r*| ≈ .10 small, .30 medium, .50 large. A correlation is not causation, and with many variables some "significant" results arise by chance, which is why a correction is offered. Always look at the scatterplot (Chapter 6) before trusting *r*.

**Reporting example (experiment data).** *"Pre-test prejudice and intergroup anxiety were positively correlated, r(298) = .45, 95% CI [.35, .53], p < .001."*

---

## 10. Compare Means (t-tests)

*Module bar: **Compare Means**.*

### 10.1 Choosing the test

Under **Test Design** choose:

* **Independent Samples** — two *different* groups (for example Contact vs Control). Needs a **Dependent Variable** (numeric) and a **Grouping Variable** with exactly two categories.
* **Paired Samples** — the *same* people measured twice (for example pre and post). Select the two measurements.

Under **Assumption** choose **Parametric (T-Tests)** or **Non-Parametric (U/W)** (Mann–Whitney *U* for independent groups, Wilcoxon signed-rank *W* for paired data). The option **Use Welch's correction if Levene's test is significant** (independent samples) makes PsyStat check the equal-variance assumption with Levene's test and switch to Welch's *t*-test, which does not assume equal variances, when the assumption is violated; otherwise the classic Student *t*-test is reported. The output states which test was used.

Press **▶ Run Comparison**.

### 10.2 What you get

* **Compare Means** table: descriptive statistics per group (*Mean, Median, SD, SE*, 95% CI, skewness, kurtosis).
* **T-Test** table: the test *Statistic*, *p-value*, and **Effect Size** (Cohen's *d*; Levene's test for equal variances for independent groups).
* **Raincloud** plot: distribution, individual points and summary per group.

### 10.3 Interpreting and reporting

Cohen's *d*: .2 small, .5 medium, .8 large. **Always report the effect size, not only *p*.**

*"Participants in the contact condition reported higher contact quality (M = 5.06, SD = 0.84) than controls (M = 3.93, SD = 0.95), t(295) = 10.93, p < .001, d = 1.27."* (These are the values you obtain from the example experiment dataset; Levene's test was not significant, so Student's *t* was used.)

If assumptions are badly violated (strong outliers, tiny samples), switch to the **Non-Parametric** option or use the **Bayesian** t-test (Chapter 22).

---

## 11. Analysis of Variance

*Module bar: **Analysis of Variance**.* Compare means across **three or more groups** (or across several factors), optionally adjusting for covariates, and analyse **repeated measures**.

### 11.1 Between-subjects designs (ANOVA, ANCOVA, MANOVA)

1. **Dependent Variables:** select **one** numeric variable for ANOVA/ANCOVA, or **two or more** for MANOVA.
2. **Fixed Factors (Categorical):** select one or more grouping variables.
3. **Covariates (Optional Metric):** add numeric control variables for ANCOVA.
4. Options: **Full Factorial (include interaction effects)**; **Also run Kruskal–Wallis** (a non-parametric one-way alternative); **Include Descriptive Statistics**.
5. Press **▶ Run Analysis of Variance**.

**You get:**

* an **ANOVA** table with effect sizes (η², partial η²), and assumption checks such as **Levene's test**;
* **pairwise comparisons** (post-hoc) with correction for multiple testing (Tukey, Games–Howell when variances are unequal, Bonferroni; and Dunn's test after Kruskal–Wallis);
* for ANCOVA, **adjusted group means** and a check of the **homogeneity of regression slopes** assumption;
* for several dependent variables, a **MANOVA** tab (Wilks' λ, Pillai's trace) with Box's M for covariance homogeneity;
* a **Raincloud** plot per group and a **Syntax** tab with reproducible **R** (`ez`) and **Python** (`pingouin`) code.

**Interpreting.** η² of about .01 is small, .06 medium and .14 large (Cohen, 1988). A significant omnibus *F* only says that *some* means differ; use the pairwise table to see which. With a significant **interaction**, interpret the simple effects, not the main effects alone.

### 11.2 Repeated measures (within-subjects)

Use this when the **same participants** were measured **three or more times**.

1. In **Repeated Measures (select 3+, in time order)** select the measurement columns **in chronological order** (for example `prejudice_pre`, `prejudice_post`, `prejudice_followup`).
2. Choose the test: **Repeated-Measures ANOVA (Parametric)**, **Friedman Test (Non-Parametric)**, or **Both**.
3. Press **▶ Run Repeated Measures Test**.

**You get:** the within-subjects *F* test with **Mauchly's test of sphericity** and the **Greenhouse–Geisser** correction (with a recommendation on when the less conservative Huynh–Feldt correction is appropriate), descriptive statistics per time point, and **Bonferroni-adjusted pairwise comparisons** between time points. The **Friedman** test comes with Wilcoxon-based pairwise follow-ups. For two time points use the **Paired Samples** *t*-test in Compare Means.

> **Limitation.** The repeated-measures test analyses one group of participants. To analyse a *mixed design* (a between-subjects factor × time), run the repeated-measures test separately for each group (use **Remove Cases** in Data Management to keep one group, then **Undo**), or use a latent growth model (Chapter 16).

---

## 12. Regression

*Module bar: **Regression**.* Predict an outcome from one or more predictors.

### 12.1 Setting up

1. Choose the **Regression Model Type**:

   | Model | Outcome (Y) |
   |---|---|
   | **Linear Regression (OLS – Hierarchical & Moderation)** | numeric |
   | **Logistic Regression (Binary Y)** | two categories (0/1) |
   | **Multinomial Logistic Regression** | nominal, 3+ categories |
   | **Ordinal Logistic Regression** | ordered, 3+ categories |
   | **Polynomial Regression (Quadratic)** | numeric; curved relation |
   | **Ridge (L2)** / **Lasso (L1) Regression** | numeric; many or correlated predictors (penalised) |

2. Select the **Dependent Variable (Y)**.
3. Put predictors in the blocks. For **non-OLS** models use **Block 1** for all predictors. For **hierarchical OLS** enter controls in **Block 1**, main predictors in **Block 2**, and additional predictors or **interaction terms** in **Block 3**; the output shows how *R*² changes at each step. (Create a product term with **Multiply & Create** in Data Management; **centre** the two variables first with **Z-Score** to reduce multicollinearity.)
4. Optionally tick **Remove Outliers Before Running (Cook's distance)** and choose the cut-off: *4/N (recommended — sensitive)*, *1.0 (SPSS default — lenient)* or *0.5 (moderate)*.
5. Press **▶ Run Regression**.

### 12.2 What you get

| Tab | Contents |
|---|---|
| **📋 Assumptions** | Normality of residuals (Shapiro–Wilk), heteroscedasticity (Breusch–Pagan), independence (Durbin–Watson), **multicollinearity (VIF and tolerance)** and flagged influential cases |
| **📊 Results** | Descriptives, model summary (*R*², adjusted *R*², *F*), coefficients (*B*, *SE*, β, *t*, *p*, CI); for logistic models odds ratios, a classification table, McFadden/Nagelkerke pseudo-*R*² and the Hosmer–Lemeshow test; ordinal models add threshold (cut-point) parameters; Ridge/Lasso show penalised coefficients for standardised predictors; **Reproducible Syntax** (R and Python) at the end |
| **Obs vs Pred, Residuals, Q-Q Plot, Cook's** | Diagnostic charts |
| **📈 Diagnostics — ROC** | (Logistic models) ROC curve and AUC |

### 12.3 Interpreting

* *B* is the change in Y per unit change in the predictor, holding the others constant; β is the standardised version, comparable across predictors.
* **VIF** above 5 (and especially above 10) signals problematic collinearity.
* If the Q-Q plot or Breusch–Pagan test shows serious violations, consider a transformation, a robust approach, or a different model.
* Report *R*², the coefficients with confidence intervals, and the sample size. Example: *"Pre-test prejudice, contact quality and anxiety jointly explained 56% of the variance in post-test prejudice, R² = .56."* (experiment data, see Chapter 25).

---

## 13. Mediation Analysis

*Module bar: **Mediation Analysis**.* Test whether the effect of X on Y operates **through** a mediator M (the indirect effect *a × b*).

### 13.1 Setting up

1. **Independent Variable(s) — X:** select one or more (each is analysed in turn). Use a numeric variable; code a two-group variable as 0/1 (for example `condition_code`).
2. **Mediator (M)** and **Dependent Variable (Y):** choose from the drop-downs.
3. **Covariates** (optional) are entered in all equations.
4. Set the **Bootstrap Resamples** (5,000 or more recommended for publication) and **Confidence Level** (95%, 90% or 99%).
5. Press **▶ Run Mediation Analysis**.

### 13.2 What you get

* **Path coefficients**: *a* (X→M), *b* (M→Y controlling for X), *c′* (the direct effect) and *c* (the total effect), each with *SE*, *t*, *p*.
* **Variance explained (R²)** for each regression equation (the M equation, the full Y equation and the total-effect equation), with adjusted *R*², *F*-test, and **ΔR²** showing what the mediator adds to the outcome model.
* The **indirect effect *a × b*** with a **percentile bootstrap confidence interval** (Preacher & Hayes, 2008) and a **Sobel test** for comparison.
* A **Path Diagram** tab and a **Bootstrap Distributions** tab (the histogram of *a × b*).
* An **APA-style write-up** sentence and reproducible syntax.

### 13.3 Interpreting

The indirect effect is **significant when its bootstrap CI excludes zero**. If the indirect effect is significant and the direct effect *c′* is not, the pattern is consistent with **full** mediation; if both are significant, **partial** mediation. Remember that mediation is a *causal* claim: it is only as convincing as your design (random assignment of X, temporal order of M and Y, no omitted confounders). The **Bootstrap** method is preferred over the Sobel test because the distribution of *a × b* is not normal.

---

## 14. Categorical PCA (CATPCA)

*Module bar: **Categorical PCA (CATPCA)**.* A principal-components method that handles **mixed data** (numeric, ordinal and nominal variables) through optimal scaling.

1. **Select Variables to Reduce (Mixed Data allowed).**
2. Set the **Number of Components to Extract**.
3. Choose **Missing Value Handling**: *Drop missing rows (Listwise)* or *Impute missing as separate category*.
4. Press **▶ Run CATPCA**.

**You get:** a table of **variable scaling levels**, a **Model Summary** (eigenvalues, variance explained, Cronbach's α per component), component **loadings**, **category quantifications** and a **Biplot**.

Use CATPCA for exploration and data reduction. If your aim is to identify latent constructs, use EFA/CFA (Chapters 17–18).

---

## 15. Cluster & Profile Analysis (LCA)

*Module bar: **Cluster & Profile Analysis (LCA)**.* Find **subgroups of people** who share similar response profiles.

1. Select the **Variables (Indicators)** (numeric).
2. Choose the **Extraction Method**: **Latent Profile Analysis (Gaussian Mixture)**, **K-Means Clustering** or **Hierarchical Clustering**.
3. Set the **Number of Classes/Clusters (k)**.
4. Press **▶ Run Latent Profile / Cluster Analysis**.

**You get:** class sizes, **class profiles** (means and SDs on the original scale), a **Profile Plot**, a PCA view of the solution, a one-way ANOVA per indicator showing which variables separate the classes, and for latent profile analysis the **average posterior probabilities** of membership and a **model-fit comparison across *k*** (AIC, BIC, entropy).

**Choosing *k*.** Run the analysis for several values of *k* and compare BIC (lower is better), entropy (closer to 1 means cleaner separation; ≥ .80 is commonly desired), the smallest class size (very small classes are rarely meaningful) and, above all, the **interpretability** of the profiles. Standardise indicators on very different scales before clustering (Data Management → **Z-Score**).

---

## 16. Forecasting (Time Series) & Latent Growth Curves

*Module bar: **Forecasting (Time Series)**.* The **Select Model** option switches between two tools.

### 16.1 Exponential smoothing (single series)

Use it to project a series that was measured regularly over time (for example monthly clinic visits).

1. Choose **Exponential Smoothing (Single Series)**.
2. Choose the **Time/Date Variable** and the **Target Variable**.
3. Set **Periods to Forecast**.
4. Press **▶ Run Model**.

**You get:** **Forecast Stats** (AIC, BIC and the estimated smoothing parameters) and a **Forecast Plot** that shows the observed series, the fitted values and the forecast with its uncertainty band. The model is **Holt's linear-trend exponential smoothing**: it captures a level and a trend but **not a seasonal cycle**, so for strongly seasonal data (such as monthly series with yearly peaks) the forecast follows the trend only. Use another tool if seasonality matters. Forecast uncertainty grows with the horizon; treat long-range forecasts cautiously, and remember that statistical forecasts assume the process continues as in the past.

### 16.2 Latent Growth Curve Modeling (LGCM)

Use it to model **individual change** across **three or more waves** of the same measure.

1. Choose **Latent Growth Curve Modeling (LGCM)**.
2. In **Select Repeated Measures (T1, T2, T3… in order)** select the wave variables in time order.
3. Press **▶ Run Model**.

**You get:** the estimated **latent intercept** (average starting level) and **latent slope** (average rate of change) with their variances (do people differ in starting point or in change?) and their covariance, and a **Growth Plot** with the mean trajectory and individual trajectories. **Global fit indices are not reported for growth models in this version** (the fit statistics for models with a mean structure are not reliable); use `lavaan::growth()` in R if you need them. Growth models need adequate sample size and equally spaced waves; with only three waves a linear growth model is the realistic choice.

---

## 17. Exploratory Factor Analysis (EFA)

*Module bar: **Exploratory (EFA)**.* Discover how many underlying factors a set of items measures and which items belong to which factor.

### 17.1 Before you start

* Use items on an interval-like scale (Likert items with 5 or more points are commonly treated this way). **Reverse-score** negatively worded items first (Chapter 5.4).
* You need a reasonable sample: at least about 5–10 participants per item and ideally 200 or more.

### 17.2 Running EFA

1. **Select Items for EFA** (items only, not IDs or demographics).
2. **Number of Factors** – enter 0 to let PsyStat use the Kaiser criterion (eigenvalues > 1), or enter the number you want after examining the scree plot and parallel analysis.
3. **Extraction method:** *minres (MINRES / PAF)*, *ml (Maximum Likelihood)* or *principal (PCA)*. For factor analysis proper, prefer **minres** or **ml**; PCA is a data-reduction method, not a factor model.
4. **Rotation:** *promax*, *oblimin* (both allow the factors to correlate and are usually the sensible default in psychology), *varimax* (forces uncorrelated factors) or *none*.
5. **Suppress Loadings <** hides small loadings (for example 0.30) so the pattern is easy to read.
6. Press **▶ Run EFA**.

### 17.3 What you get

* **Assumptions & Scree** tab: the **KMO** measure of sampling adequacy and **Bartlett's test of sphericity** (checks that the items are correlated enough to factor), a **scree plot** and **parallel analysis**.
* **Matrix & Corr** tab: *Total Variance Explained*, the **factor loadings and communalities**, **internal consistency** (Cronbach's α and McDonald's ω) per factor, and the **factor correlation matrix** (for oblique rotations).
* **Syntax** tab: reproducible R (`psych`) and Python (`factor_analyzer`) code.

### 17.4 Interpreting

* **KMO** ≥ .60 is acceptable (≥ .80 good); Bartlett's test should be significant.
* Decide the number of factors using **several** criteria (scree, parallel analysis, interpretability), not the Kaiser rule alone, which tends to over-extract.
* Loadings ≥ .40 are generally meaningful. An item that loads weakly everywhere, or substantially on two factors (**cross-loading**), is a candidate for removal. Re-run the analysis after removing an item.
* Exploratory results should be **confirmed in a new sample** with CFA (Chapter 18); do not run EFA and CFA on the same data and call the CFA a confirmation.

---

## 18. Confirmatory Factor Analysis (CFA)

*Module bar: **Confirmatory (CFA)**.* Test a measurement model that you specify in advance.

### 18.1 Writing the model

Use **lavaan-style syntax** in the model box. `=~` means "is measured by":

```
Anxiety    =~ item1 + item2 + item3
Depression =~ item4 + item5 + item6
```

Each latent factor needs at least three indicators for a unique solution (two can work if factors are correlated and other constraints exist). Correlated errors (use sparingly and only with a theoretical reason) are written `item1 ~~ item2`. Latent factors are allowed to correlate by default. The first indicator of each factor is fixed to 1 to give the factor a scale.

### 18.2 Options

| Option | Choices | Notes |
|---|---|---|
| **Estimator** | **ML** – Maximum Likelihood (normal-theory SE) · **MLR** – ML with robust (sandwich) SE and scaled χ² | Use **MLR** when data depart clearly from multivariate normality (skewed or heavy-tailed items). The point estimates are the same as ML; the standard errors are Huber–White sandwich estimates and the test statistic is the Satorra–Bentler scaled χ², with correspondingly scaled CFI, TLI, NFI and RMSEA. MLR uses complete cases only. |
| **Data metric** | **Standardised** – z-scored data (unstandardised = Std.all) · **Raw data** – estimates in the original scale of the instrument | Choose *Raw* if you want unstandardised loadings in the units of your items (the first loading of each factor is fixed to 1). Std.lv and Std.all are always reported, computed from the model-implied covariance matrix. Fit statistics are the same in both metrics. |
| **Missing data method** | **Listwise deletion** (default; all fit indices) · **FIML** (uses all rows; global fit indices unavailable) | With FIML, estimates and standard errors use every observation, but χ², CFI, TLI, RMSEA and SRMR are not computed in this version; run listwise to obtain them. FIML cannot be combined with MLR (MLR then uses listwise deletion). |
| **My indicators are ordinal (Likert ≤ 5 pts) — show warning** | tick box | Shows a caution that ML on ordinal items with few categories can distort loadings and fit (see 18.4). |

Press **▶ Run CFA**.

### 18.3 What you get

* **Model Information** (estimator, data metric, sample size) and the **Analysis Note** stating how missing data were handled.
* **Model Fit Indices**: χ² (df, *p*), CFI, TLI, NFI, GFI, RMSEA, SRMR, AIC and BIC, each with a guideline (CFI/TLI ≥ .95 excellent, ≥ .90 acceptable; RMSEA ≤ .06 excellent, ≤ .08 acceptable; SRMR ≤ .08; Hu & Bentler, 1999). With MLR the scaled versions are shown and labelled.
* **Parameter Estimates**: loadings (unstandardised, Std.lv, Std.all), variances and covariances with SE, *z* and *p*; the first loading of each factor is fixed and has no SE.
* **Construct Reliability and Validity**: Cronbach's α, McDonald's ω, **composite reliability (CR)** and **average variance extracted (AVE)**, with the **Fornell–Larcker** check of discriminant validity (the square root of each construct's AVE should exceed its correlations with other constructs).
* **Modification Indices** tab: which fixed parameters (for example a residual covariance) would most improve fit.

### 18.4 Interpreting and reporting

* Judge fit with **several indices together**. χ² is nearly always significant in large samples and should not be the only criterion.
* Standardised loadings of .50–.70 or higher indicate adequate indicators; AVE ≥ .50 and CR ≥ .70 are common benchmarks (Fornell & Larcker, 1981).
* **Modification indices** are data-driven. Adding paths because an index is large capitalises on chance; make only changes you can justify theoretically, and say so in the paper.
* **Ordinal data.** Likert items with five or fewer categories are, strictly, ordinal. ML (and MLR) treat them as continuous, which works acceptably with five or more categories and moderate skew, but underestimates loadings otherwise. Dedicated ordinal estimators (WLSMV with polychoric correlations) are not available in PsyStat; use R's `lavaan` for those.
* **Reporting example.** *"The four-factor model fitted the data well, χ²(84) = 84.0, p = .48, CFI = 1.00, TLI = 1.00, RMSEA = .002, SRMR = .04."* (example survey data; see Chapter 25).

---

## 19. Structural Equation Modeling (SEM)

*Module bar: **SEM (Graph & Syntax)**.* Combine measurement models with **structural paths** between latent and observed variables.

### 19.1 Two ways to specify the model

**A. Interactive Builder** (tab *Interactive Builder*): draw the model on a canvas.

* **✚ Add Latent** adds a latent (circle) variable; **✚ Add Observed** adds an observed (rectangle) variable.
* **➚ Draw Path** connects two variables with a regression path or loading (click the source, then the target); **↔ Covariance** adds a covariance; **✖ Delete** removes an element.
* **Generate Syntax ↓** converts the drawing into syntax in the *Syntax & Results* tab.

**B. Syntax** (tab *Syntax & Results*): write lavaan-style syntax directly:

```
# measurement part
SI =~ si1 + si2 + si3 + si4 + si5
IC =~ ic1 + ic2 + ic3 + ic4
PR =~ pr1 + pr2 + pr3
# structural part
PR ~ IC + SI
```

**Load Example** inserts a ready-made model.

### 19.2 Running

Choose the **Missing data method**: **Listwise deletion** (default) or **FIML** (uses all rows; global fit indices are not reported under FIML), tick the **ordinal indicators warning** if relevant, and press **▶ Estimate Structural Model**.

### 19.3 Output

The **SEM Results** tab has the same sections as CFA (model information, fit indices, parameter estimates with Std.lv/Std.all), including the **structural regressions** (paths) with standardised coefficients. See 18.3 and 18.4 for interpretation.

### 19.4 Good practice

Specify the model from theory *before* looking at the data; check the measurement model with CFA first; avoid adding many paths to improve fit; and report the full model with all fit indices. A well-fitting model is *consistent with* your theory but does not prove it, because equivalent models can fit equally well. Variables are z-scored before fitting, so unstandardised estimates equal Std.all in this module.

---

## 20. Network Analysis

*Module bar: **Network Analysis**.* Map the relationships among symptoms, items or traits as a **network**: variables are *nodes* and (partial) correlations are *edges*.

### 20.1 Choosing variables and method

1. **Select Variables for Network** (numeric; typically 5–40 items or scores). **Do not include a total score together with the items it is built from**: the total is perfectly predictable from its items, which makes partial correlations undefined (the app stops with an explanation for the partial-correlation methods and warns for EBICglasso).
2. **Estimation method:**
   * **EBICglasso (recommended)** – regularised partial correlations (graphical lasso; the model is selected with the Extended BIC, γ = 0.5 by default; Friedman et al., 2008; Foygel & Drton, 2010; Epskamp & Fried, 2018). Produces sparse, interpretable networks. Needs **scikit-learn**.
   * **Partial correlations – significant edges** – unregularised partial correlations keeping an edge only if its **adjusted** *p*-value is below α. Choose **Bonferroni**, **Holm** or **Benjamini–Hochberg (FDR)** correction.
   * **Partial correlations – unregularised** – all partial correlations filtered only by the minimum edge weight.
   * **Correlation network (zero-order)** – raw correlations; useful for comparison, but dense.
3. **Correlation type:** **Pearson**, **Spearman (rank-based)** or **Nonparanormal (normal scores)**. Use Spearman or Nonparanormal for skewed or ordinal (Likert) data.
4. **Minimum |edge weight| to display** hides weak edges (defaults: 0 for EBICglasso and significance-tested networks; 0.15 for unregularised partial correlations; 0.30 for the correlation network).
5. Optionally tick **Bootstrap accuracy & stability analysis** and set the number of bootstrap samples.
6. Choose the **Graph Layout** (spring/Fruchterman–Reingold, Kamada–Kawai, circular).

### 20.2 Appearance controls

* **Node colours:** *Auto – detected communities* colours the nodes by community found with the greedy modularity algorithm. *Custom – my own subscales* lets you assign variables to **named subscales or constructs** (button **Define Subscales & Colours…**), and pick a colour for each. Variables that you leave out are grey ("Unassigned").
* **Positive / negative edge colour** (default green / red) and **Draw negative edges dashed** (so that sign remains visible in black-and-white print).
* **Node label font size**, **Node label colour** (*Auto, Black, White*) and **Node size** (30–250 %).
* **Reset colours to default** restores the defaults.

### 20.3 What you get

* **Network Metrics:** the estimator, correlation type, *N*, number of edges and sparsity; node **strength**, **expected influence** (signed), **degree**, **betweenness** and **closeness** (weighted); the **edge-weight list**, and a caution if variables are (nearly) collinear.
* **Network Plot:** green solid lines are **positive** and red dashed lines **negative** relations. **Line width and opacity grow with the absolute weight** (relative to the strongest edge). Node colour shows the community or subscale, and a legend explains everything.
   * **Move nodes** by dragging them with the mouse.
   * For large networks the plot **scrolls** (mouse wheel for vertical, **Shift + wheel** for horizontal, **Ctrl + wheel** to zoom); use the **− / + / Fit** buttons to zoom.
   * **⤢ Open in Window** shows the network in a separate resizable window.
   * **💾 Save 300 DPI** saves a PNG or TIFF at 300 DPI, or a vector SVG/PDF.
* **Centrality Charts** and **Bridge Analysis** (which nodes connect communities).
* If bootstrapping was selected: **Bootstrap Summary** (centrality **CS-coefficients** and edge-weight confidence intervals) and **Stability Plots**.

### 20.4 Interpreting

* An edge is the association between two variables **after controlling for all the others**. It is not evidence of causation or its direction.
* **Centrality** identifies nodes that are strongly connected, but centrality indices can be unstable. Interpret them only if the bootstrap shows adequate stability: the **CS-coefficient should be above 0.25 and preferably above 0.50** (Epskamp, Borsboom & Fried, 2018).
* For regularised estimates the bootstrap intervals describe the **accuracy** of edge weights; they are not significance tests, and an interval that includes zero does not mean that the edge is absent.
* The bootstrap for EBICglasso fits each resample on a narrow window of tuning values around the original solution to save time; this is a documented approximation. Bootstrapping with many variables can take several minutes. A **Cancel** button is provided.
* Networks from small samples are unreliable. As a rough guide you want considerably more participants than the number of edges you hope to estimate, and you should report the sample size and sparsity.

---

## 21. Power Analysis

*Module bar: **Power Analysis**.* Plan sample size (or judge a finished study) in the style of G*Power.

### 21.1 Choosing a design

PsyStat covers ten designs: **Independent Samples *t*-test**, **Paired Samples *t*-test**, **One-Way ANOVA**, **Two-Way / Factorial ANOVA** (main effects of Factor A or B, or the A × B interaction), **Multiple Regression – Omnibus *R*²**, **Multiple Regression – *R*² Increase / Interaction**, **Logistic Regression** (continuous predictor, binary outcome), **Pearson Correlation**, **Chi-Square test**, and **Mediation** (indirect effect; see the note in the module).

### 21.2 Three modes

| Mode | Question | You supply | You get |
|---|---|---|---|
| **A Priori — Required N** | *How many participants do I need?* | Effect size, α, target power (1 − β) | Required N |
| **Post-Hoc — Achieved Power** | *What power did my study have?* | Effect size, α, the N you have | Achieved power |
| **Sensitivity — Min Detectable Effect** | *What is the smallest effect I could detect?* | N, α, target power | Minimum detectable effect |

Choose **Two-tailed** or **One-tailed** (a one-tailed test needs a strong prior reason). Enter the design-specific inputs (number of groups, number of predictors, etc.), then press **▶ Calculate Power**.

### 21.3 Output

A **Power Calculation** tab (a G*Power-style summary of inputs and results, plus Cohen's 1988 conventions for small/medium/large effects) and a **Power Curve** tab showing power as a function of sample size.

### 21.4 Good practice

* Base the **effect size on prior literature or the smallest effect of practical interest**, not on a guess of "medium". Effect sizes from small pilot studies are noisy and usually too large.
* **Be careful with Post-Hoc mode.** "Observed power" computed from a completed study's own result only restates its *p*-value and is not informative. Use Post-Hoc with an effect size taken from the *literature* (or the smallest effect of interest), or use the sensitivity mode.
* Aim for at least 80 % power; many journals now expect 90 % for confirmatory work.
* Plan for **attrition** and exclusions by recruiting more than the computed N.

---

## 22. Bayesian Analysis

*Module bar: **Bayesian Analysis**.* Quantify evidence **for and against** a hypothesis, including evidence for the *null*. Reporting follows APA 7 and the guidelines of van Doorn et al. (2021).

### 22.1 Analyses available

*Independent-samples t-test*, *Paired t-test*, *Pearson correlation*, *Linear regression*, *ANOVA* and *Mediation*. Select the analysis type, then the variables (drag them from the *Available Variables* list or use the lists), and press the run button.

### 22.2 Priors

* **Cauchy scale *r*** (t-tests): the prior on the standardised effect size δ is Cauchy(0, *r*). The default **r = 0.707** is the JASP/BayesFactor default ("medium" prior). Smaller values assume smaller effects; larger values (for example 1.0) assume larger effects.
* **Stretched-beta width κ** (correlation): the prior on ρ is a stretched Beta(1/κ, 1/κ) on (−1, 1) (Ly, Verhagen & Wagenmakers, 2016). **κ = 1** is the uniform prior (JASP default).
* Regression and ANOVA Bayes factors use the **BIC approximation** (a unit-information prior; Raftery, 1995; Wagenmakers, 2007).
* **Mediation** uses a **Bayesian bootstrap** (Rubin, 1981): a posterior under a non-informative prior, with no Bayes factor.

Always check how sensitive the conclusion is to the prior by re-running with a different *r* or κ, and report that.

### 22.3 What the tables show

Each analysis shows **two tables**:

1. **Descriptive & Frequentist Reference** – sample statistics, the frequentist *t*, *p*-value, confidence interval and effect size. These are marked with an asterisk and **are shown for comparison only**; they play no role in the Bayesian inference.
2. **Bayesian Inference** – the **prior**; the **posterior median and mean** of the effect (δ or ρ); the **95% credible interval (CrI, highest-density)**; **BF₁₀**, **BF₀₁ = 1/BF₁₀** and an evidence label.

Beneath the table a **note** gives the prior, the method, a plain-language statement ("The data are 5.1 times more likely under H₀ than under H₁") and, for correlations, a suggested reporting sentence.

### 22.4 Reading Bayes factors

| BF₁₀ | Evidence for H₁ | | BF₁₀ | Evidence for H₀ (use BF₀₁) |
|---|---|---|---|---|
| 1–3 | Anecdotal | | 1/3–1 | Anecdotal / inconclusive |
| 3–10 | Moderate | | 1/10–1/3 | Moderate |
| 10–30 | Strong | | 1/30–1/10 | Strong |
| 30–100 | Very strong | | 1/100–1/30 | Very strong |
| > 100 | Decisive | | < 1/100 | Decisive |

(Jeffreys, 1961; Kass & Raftery, 1995.) A BF near 1 means the data do not discriminate between the hypotheses; this is **not** evidence for either. A credible interval is read directly: *given the prior and the data, there is a 95% probability that the effect lies in this interval.*

### 22.5 Reporting example (illustrative values)

*"A Bayesian independent-samples t-test with a default Cauchy prior (r = 0.707) provided moderate evidence for the null hypothesis of no group difference, BF₀₁ = 5.09 (BF₁₀ = 0.197); the posterior median of δ was −0.03, 95% CrI [−0.36, 0.32]. A prior sensitivity check with r = 1.0 gave BF₀₁ = 7.00."*

---

## 23. Exporting, saving and settings

### 23.1 Exporting results

Every results panel has a **💾 Export to HTML/Word** button. It saves the **currently displayed tab** as:

* **.html** – opens in any browser and can be pasted into Word or Google Docs with the formatting intact; or
* **.doc** – a Word-compatible file (HTML-based), which opens directly in Microsoft Word, LibreOffice and Pages.

Tables keep their APA styling. Charts are exported as image files with **Save 300 DPI** (a button in the chart window).

### 23.2 APA Mode

At the top right of the window the **Exploratory Mode / APA Mode** switch controls the look of result tables.

* **Exploratory Mode** (default): colour highlights and badges make the results quick to scan.
* **APA Mode**: black-and-white, journal-style tables (Times New Roman, only *p*-values in bold, no colour). Use it when you want to paste tables straight into a manuscript.

### 23.3 Decimal places

**Settings → Decimal Formatting** sets 2, 3 or 4 decimals for all tables. APA recommends two decimals for most statistics and three for *p*-values; *p* is shown without a leading zero (*p* = .032, *p* < .001).

### 23.4 Light and dark mode

**Settings → Toggle Light/Dark Mode**.

---

## 24. Autosave and crash recovery

PsyStat saves a copy of your dataset in the background **every two minutes** into a **private folder in your operating system's temporary directory**; nothing is uploaded anywhere. If the program closes unexpectedly (a crash, a power cut), the next time you start PsyStat it offers to **recover your last session**.

* Autosave is a safety net, **not a substitute for saving**. Use **File → Save Dataset As…** regularly.
* If autosave fails (for example because the disk is full), PsyStat shows a **warning dialog** and a status-bar message. Free up disk space and save your data manually.
* **Shared computers (labs).** Each user account has its own autosave folder. Autosave files are removed when you close the program normally, but **after a crash they remain** until you recover or discard them. If you used a shared or generic account for sensitive data, discard the recovery offer on the next start and delete your dataset files when you are finished.

---

## 25. Tutorials with the example datasets

PsyStat ships with **four simulated datasets** (File → Load Sample Data). They contain no real participants. The tutorials below walk through every module and tell you what to expect. A full description of every variable is in the **Examples Guide** (Help → Example Datasets Guide).

| Dataset | Description | Used in |
|---|---|---|
| **Experiment Example (N = 300)** | Contact vs Control experiment with pre-test, post-test and follow-up | Tutorials A1–A10 |
| **Survey Example (N = 250)** | 15 Likert items, 4 constructs | Tutorials B1–B7 |
| **CVI Example** | 12 items rated by 6 experts | Tutorial C1 |
| **Time Series Example** | 60 months of counselling visits | Tutorial C2 |

### A. Experiment dataset (File → Load Sample Data → Experiment Example)

**A1. Descriptives.** *Descriptives & Crosstabs* → select `prejudice_pre`, `prejudice_post`, `contact_quality_post` → **Generate Descriptives** → **Run**. Check that the means are about 4.0, 4.0 and 4.5 and that `contact_quality_post` has two missing values per hundred cases (about 2%).

**A2. Independent-samples t-test.** *Compare Means* → **Independent Samples**, *Parametric* → Dependent = `contact_quality_post`, Grouping = `condition` → **Run Comparison**.
*Expect:* Contact M = 5.06, SD = 0.84 (*n* = 147); Control M = 3.93, SD = 0.95 (*n* = 150); *t*(295) = 10.93, *p* < .001, Cohen's *d* = 1.27 (a very large effect). Levene's test is not significant (*p* = .207), so the classic Student *t*-test is used.

**A3. Paired-samples t-test, and why subgroups matter.** *Compare Means* → **Paired Samples** → `prejudice_pre` vs `prejudice_post` for **all** participants: *t* ≈ 1.03, *p* = .31, *d* ≈ 0.06 — no overall change. The two conditions pull in opposite directions, so the average hides the effect. Now go to *Data Management*, **Remove Cases** where `condition == Control`, and repeat: for the Contact group, *t*(149) ≈ 3.07, *p* = .003 (prejudice fell from about 3.96 to 3.78). Press **↩ Undo** to restore the removed rows.

**A4. One-way ANOVA.** *Analysis of Variance* → Dependent = `cooperation_post`, Fixed Factor = `condition` → **Run**. *Expect:* *F*(1, 290) = 24.2, *p* < .001, partial η² ≈ .08 (Contact cooperated more). Then use `age` as the dependent variable and `gender` (three groups) as the factor to see a **null result** with post-hoc tests: *F* = 0.24, *p* = .79.

**A5. ANCOVA.** Dependent = `prejudice_post`, Fixed Factor = `condition`, **Covariate** = `prejudice_pre`. *Expect:* the effect of condition remains significant after adjusting for pre-test prejudice, *F*(1, 297) ≈ 21.6, *p* < .001, and the pre-test is a very strong covariate (*F* ≈ 199).

**A6. Repeated-measures ANOVA.** First keep only the Contact group (A3). *Analysis of Variance* → **Repeated Measures** → select `prejudice_pre`, `prejudice_post`, `prejudice_followup` in this order → **Repeated-Measures ANOVA** → **Run**.
*Expect:* *F*(2, 286) ≈ 22.2, *p* < .001; Mauchly's *W* = .86 (*p* < .001, so sphericity is **violated**) and Greenhouse–Geisser ε = .87 (the corrected *p* is still < .001). Means: 3.96 → 3.78 → 4.13: prejudice drops after contact and then **rebounds** at follow-up. (*N* = 144 participants with complete data at all three times.)

**A7. Hierarchical regression.** *Regression* → **Linear Regression (OLS)**, Dependent = `prejudice_post`, **Block 1** = `prejudice_pre`, **Block 2** = `contact_quality_post` and `intergroup_anxiety_post` → **Run Regression**.
*Expect:* Block 1 *R*² = .396; adding Block 2 gives *R*² = .563, Δ*R*² = .167, *F* change(2, 289) = 55.4, *p* < .001. The largest VIF is 1.30, so there is no collinearity problem.

**A8. Mediation.** *Mediation Analysis* → X = `condition_code`, M = `contact_quality_post`, Y = `prejudice_post` → **Run** with at least 2,000 bootstrap resamples.
*Expect:* *a* = 1.14 (*p* < .001); *b* = −0.36 (*p* < .001); total effect *c* = −0.37 (*p* < .001); direct effect *c′* = 0.04 (*p* = .71); **indirect effect *a × b* ≈ −0.41 with a bootstrap CI that excludes zero**. The pattern is consistent with *full mediation*: contact reduces prejudice by improving the quality of contact. (The experiment was randomised, which supports the causal reading of X→M; the M→Y link is not randomised.)

**A9. Moderation chart.** *Data Visualization* → **Moderation Chart** → X = `contact_quality_post`, Y = `prejudice_post`, Moderator = `group_identification_pre` → **Generate Graph**. *Expect:* the negative slope of contact quality on prejudice is steeper for participants with high group identification (the interaction coefficient is about −0.10, *p* = .012; simple slopes ≈ −0.18 at −1 SD and ≈ −0.35 at +1 SD).

**A9b. Correlation.** *Correlation* → select the `…_pre` variables (`prejudice_pre`, `contact_quality_pre`, `intergroup_anxiety_pre`, `cooperation_pre`, `group_identification_pre`) → **Pearson**, listwise. *Expect:* prejudice and anxiety correlate positively (*r* = .45), prejudice and prior contact quality negatively (*r* = −.31).

**A10. Latent profiles and growth.** *Cluster & Profile Analysis* → select the four post-test variables → **Latent Profile Analysis**, *k* = 2 → **Run**: two classes (about 47% / 53%) with different profiles. *Forecasting* → **Latent Growth Curve Modeling** → select `prejudice_pre`, `prejudice_post`, `prejudice_followup` → **Run Model** to see the average intercept and slope and the individual trajectories. (Fit indices are not reported for growth models in this version.)

### B. Survey dataset (File → Load Sample Data → Survey Example)

**B1. Reverse-score first.** Item `si5` is reverse-worded. *Data Management* → **Reverse-Score Variables** → select `si5`, Min = 1, Max = 5 → **Reverse Selected** (keep *Create as new variable* ticked and use the new `si5` name, or overwrite).

**B2. Reliability.** *Item Analysis & CVI* → select `si1`–`si5` → **Run Item Analysis**. *Expect:* α ≈ .75 for Social Identity with `si5` reverse-scored. Re-run **without** reversing `si5` to see the damage: α drops to about .40, and `si5` has a negative item-rest correlation. Repeat for the other scales: Intergroup Contact (`ic1`–`ic4`) α ≈ .80, Prejudice (`pr1`–`pr3`) α ≈ .74, Wellbeing (`wb1`–`wb3`) α ≈ .67 (acceptable but modest).

**B3. Exploratory factor analysis.** *Exploratory (EFA)* → select all 15 items → Number of Factors = 4 (or 0 for the Kaiser rule), **minres**, **oblimin** → **Run EFA**. *Expect:* KMO = .77, Bartlett's test significant (χ²(105) ≈ 801, *p* < .001), four eigenvalues above 1 (3.48, 2.56, 1.57, 1.31), and a clean four-factor pattern that matches the intended scales.

**B4. Confirmatory factor analysis.** *Confirmatory (CFA)* → enter the model:

```
SI =~ si1 + si2 + si3 + si4 + si5
IC =~ ic1 + ic2 + ic3 + ic4
PR =~ pr1 + pr2 + pr3
WB =~ wb1 + wb2 + wb3
```

→ **ML**, *Standardised*, **Listwise** → **Run CFA**. *Expect* (*N* = 207 complete cases): χ²(84) = 84.04, *p* = .48; CFI = 1.00, TLI = 1.00, RMSEA = .002, SRMR = .043. (Fit this good is typical of data simulated from exactly this model; real data rarely fit this well.) Then try **FIML**: the model now uses all 250 participants, but no global fit indices are shown. Try **MLR** and **Raw data** and compare the standard errors and the unstandardised loadings.
Note the warning that 17% of rows are lost under listwise deletion: with 15 items and about 1% missing per item, many participants miss at least one response. This is why FIML is useful.

**B5. SEM.** *SEM (Graph & Syntax)* → *Syntax & Results*:

```
SI =~ si1 + si2 + si3 + si4 + si5
IC =~ ic1 + ic2 + ic3 + ic4
PR =~ pr1 + pr2 + pr3
WB =~ wb1 + wb2 + wb3
PR ~ IC + SI
WB ~ IC + PR
```

→ **Estimate Structural Model**. *Expect* (standardised paths, *N* = 207): intergroup contact predicts **lower prejudice** (β ≈ −.49, *p* < .001); social identity predicts slightly **higher** prejudice (β ≈ .25, *p* ≈ .01); intergroup contact predicts higher wellbeing (β ≈ .24, *p* ≈ .02); prejudice predicts lower wellbeing, a weaker effect that is right at the border of significance (β ≈ −.21, *p* ≈ .05). It is a useful reminder that a path near *p* = .05 should be reported with its confidence interval and interpreted cautiously.

**B6. Network.** *Network Analysis* → select all 15 items → **EBICglasso**, **Spearman** → **Run**. *Expect:* about 31 of 105 possible edges (30%) and **four communities** (Q ≈ .64) that match the four scales. Try **Define Subscales & Colours…** to colour nodes by your own scales and **Save 300 DPI** for a figure. Tick **Bootstrap** (100 samples) and read the **CS-coefficients**.

**B7. Bayesian correlation and CATPCA.** First compute scale scores: *Data Management* → select `si1`–`si5` → name `SI_mean` → **Mean** (and likewise `IC_mean`, `PR_mean`, `WB_mean`; remember that `si5` must be reverse-scored first). Then *Bayesian Analysis* → **Pearson correlation**. For `SI_mean` and `WB_mean` (sample *r* ≈ .05) you should find **evidence for the null**: BF₀₁ ≈ 9.4 (BF₁₀ ≈ 0.11), moderate evidence that the two are unrelated. For `IC_mean` and `PR_mean` (sample *r* ≈ −.33) the evidence for a relation is overwhelming (BF₁₀ above 100,000). Change the stretched-beta width κ to see how the Bayes factor responds to the prior. *Categorical PCA* → select the 15 items, 3 components, to view the component structure and biplot.

### C. Bonus datasets

**C1. Content validity.** *Item Analysis & CVI* → CVI tool → select `expert1`–`expert6`, set the threshold to **2.5** → **Calculate I-CVI**. *Expect:* items 1–6, 8 and 9 reach I-CVI = 1.00 and item 7 reaches .83 (all acceptable with six experts, where I-CVI ≥ .78 is the usual benchmark); items 10, 11 and 12 have I-CVI of .33, .50 and .33 and should be revised or dropped. S-CVI/Ave = .83 (below the .90 benchmark, because of those three items); S-CVI/UA = .67.

**C2. Forecasting.** *Forecasting* → **Exponential Smoothing** → Time = `month`, Target = `counselling_visits`, Periods = 12 → **Run Model**. The series rises with a yearly cycle; the fitted model is Holt's linear trend and does **not** model the seasonal cycle, so examine the plot: the forecast follows the trend but not the seasonal peaks, a typical lesson in why seasonality should be modelled when it is strong.

**C3. Power.** *Power Analysis* → **Independent Samples t-test**, **A Priori**, two-tailed, *d* = 0.5, α = .05, power = .80 → *Expect* **64 participants per group (128 in total)**. Change *d* to 0.3 to see how quickly the required sample grows (176 per group, 352 in total).

---

## 26. Troubleshooting & FAQ

**The app does not start / the window never appears.**
Wait up to 30 seconds on the first start. On Windows, make sure you clicked *More info → Run anyway* in SmartScreen; on macOS, use *right-click → Open* the first time. If you run from source, check that Python is 3.11 or 3.12 and that `pip install -r requirements.txt` finished without errors.

**"Missing dependency: please run pip install …" (running from source).**
Install the named package in the same Python environment you use to start PsyStat (`pip install <name>`), then restart. Packaged downloads already include every dependency.

**My CSV opens with all variables in one column / strange characters.**
The file may use semicolons or a different encoding. Open it in Excel and save it as *CSV UTF-8*, or re-export from your source program.

**A numeric variable is treated as text.**
It probably contains a non-numeric entry (for example "N/A" or a space). Open the Dataset tab, fix or clear the offending cell, or use **Duplicate & Convert Variable → Force to Numeric**.

**An analysis says "not enough complete cases".**
Listwise deletion removed too many rows. Check the *Missing Values* column of Variable Info, remove variables that are mostly empty, or use FIML in CFA/SEM.

**CFA/SEM shows warnings about ordinal data.**
Likert items with five or fewer categories are ordinal. ML/MLR results are an approximation; see Chapter 18.4.

**CFI / RMSEA are not shown.**
You chose FIML. Global fit indices are not available under FIML in this version; run listwise deletion to obtain them. They are also not reported for latent growth models.

**The network plot is cut off or too small.**
Use the scroll wheel (Ctrl + wheel to zoom), the **Fit** button, or **⤢ Open in Window** for a larger view. Lower **Node size** for many variables.

**EBICglasso says scikit-learn is required.**
Install it (`pip install scikit-learn`) when running from source. Packaged downloads include it.

**Results differ slightly from SPSS / R / JASP.**
Small differences can arise from different defaults (for example Welch versus Student *t*, the correlation estimator, or rounding). Check the *Note* under each table for the exact method. If a difference is large, please report it (see below).

**How do I report a problem?**
Open an issue at <https://github.com/terysetn02-source/psystat/issues> and include: your operating system and PsyStat version (Help → About), what you did, what you expected, what happened, and (if possible) a small example dataset with identifying information removed. Never post real participant data.

**Is my data sent anywhere?**
No. PsyStat runs entirely on your computer and does not upload data or require an internet connection.

---

## 27. Statistical notes, limitations and references

### 27.1 Principles

* **Tables report what was computed.** Each result states the method and, when relevant, how missing data were handled (listwise deletion, FIML), so the analysis can be reproduced.
* **p-values and effect sizes.** Report effect sizes with confidence intervals alongside *p*-values. A significant *p* does not imply a large or important effect.
* **Multiple testing.** When many tests are run, use the correction options (Correlation, ANOVA post-hoc, Network) or pre-register your analyses.
* **Assumptions.** Check them (the tables and charts help) but remember that tests of assumptions are themselves sensitive to sample size.

### 27.2 Known limitations of this version

* No multilevel or mixed-effects models, no multiple-group CFA/SEM, no imputation, and no ordinal/WLSMV estimators.
* Global fit indices are not available under FIML or for latent growth models.
* Forecasting uses Holt's linear exponential smoothing without a seasonal component.
* CATPCA is implemented with the *prince* library (MCA for categorical, FAMD for mixed and PCA for numeric variables) rather than with full optimal scaling.
* The network bootstrap for EBICglasso uses a restricted tuning-parameter window for speed.
* Results are validated by tests against reference implementations and simulations, but no software is bug-free; verify critical results independently.

### 27.3 References

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B, 57*, 289–300.

Bentler, P. M. (1990). Comparative fit indexes in structural models. *Psychological Bulletin, 107*, 238–246.

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum.

Epskamp, S., Borsboom, D., & Fried, E. I. (2018). Estimating psychological networks and their accuracy: A tutorial paper. *Behavior Research Methods, 50*, 195–212.

Epskamp, S., & Fried, E. I. (2018). A tutorial on regularized partial correlation networks. *Psychological Methods, 23*, 617–634.

Field, A. (2013). *Discovering statistics using IBM SPSS Statistics* (4th ed.). Sage.

Foygel, R., & Drton, M. (2010). Extended Bayesian information criteria for Gaussian graphical models. *Advances in Neural Information Processing Systems, 23*, 604–612.

Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation models with unobservable variables and measurement error. *Journal of Marketing Research, 18*, 39–50.

Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics, 9*, 432–441.

Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance structure analysis. *Structural Equation Modeling, 6*, 1–55.

Jeffreys, H. (1961). *Theory of probability* (3rd ed.). Oxford University Press.

Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the American Statistical Association, 90*, 773–795.

Kline, R. B. (2016). *Principles and practice of structural equation modeling* (4th ed.). Guilford Press.

Ly, A., Verhagen, J., & Wagenmakers, E.-J. (2016). Harold Jeffreys's default Bayes factor hypothesis tests. *Journal of Mathematical Psychology, 72*, 19–32.

Lynn, M. R. (1986). Determination and quantification of content validity. *Nursing Research, 35*, 382–385.

Polit, D. F., & Beck, C. T. (2006). The content validity index: Are you sure you know what's being reported? *Research in Nursing & Health, 29*, 489–497.

Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies for assessing and comparing indirect effects in multiple mediator models. *Behavior Research Methods, 40*, 879–891.

Raftery, A. E. (1995). Bayesian model selection in social research. *Sociological Methodology, 25*, 111–163.

Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., & Iverson, G. (2009). Bayesian t tests for accepting and rejecting the null hypothesis. *Psychonomic Bulletin & Review, 16*, 225–237.

Rubin, D. B. (1981). The Bayesian bootstrap. *The Annals of Statistics, 9*, 130–134.

Satorra, A., & Bentler, P. M. (1994). Corrections to test statistics and standard errors in covariance structure analysis. In A. von Eye & C. C. Clogg (Eds.), *Latent variables analysis* (pp. 399–419). Sage.

van Doorn, J., van den Bergh, D., Böhm, U., et al. (2021). The JASP guidelines for conducting and reporting a Bayesian analysis. *Psychonomic Bulletin & Review, 28*, 813–826.

Wagenmakers, E.-J. (2007). A practical solution to the pervasive problems of *p* values. *Psychonomic Bulletin & Review, 14*, 779–804.

West, S. G., Finch, J. F., & Curran, P. J. (1995). Structural equation models with nonnormal variables. In R. H. Hoyle (Ed.), *Structural equation modeling* (pp. 56–75). Sage.

---

## 28. How to cite PsyStat

If you use PsyStat in published work, please cite it, and also cite the statistical methods you used.

> Setiawan, T. (2026). *PsyStat: A statistical analysis application for psychology and the social sciences* (Version 1.2.0) [Computer software]. https://github.com/terysetn02-source/psystat

A machine-readable citation file (`CITATION.cff`) is included in the repository; GitHub shows a **"Cite this repository"** button from it.

PsyStat is released under the **MIT License**. It builds on the open-source projects NumPy, pandas, SciPy, statsmodels, scikit-learn, matplotlib, pingouin, semopy, factor_analyzer, prince, networkx and Qt (PyQt6), each under its own licence.
