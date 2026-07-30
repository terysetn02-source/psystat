# Example Datasets Guide

Two sample datasets are included in this folder to help you explore PsyStat's
features immediately after installation. Both datasets are synthetic — they do
not contain real participant data.

---

## 1. `experiment_example.csv` — Intergroup Contact Experiment (N = 300)

### Research Context

This dataset simulates a controlled intergroup contact experiment investigating
whether structured contact between social groups reduces prejudice. Participants
were assigned to either a **Contact** condition (received an intergroup contact
intervention) or a **Control** condition (no intervention). All psychological
variables were measured before (Pre) and after (Post) the intervention.

### Variable Codebook

| Variable | Type | Description |
|---|---|---|
| `SubjectID` | String | Participant identifier (C000–C299) |
| `Condition` | Categorical | Experimental condition: `Control` or `Contact` |
| `Pre_Prejudice` | Continuous (1–7) | Prejudice score before intervention |
| `Post_Prejudice` | Continuous (1–7) | Prejudice score after intervention |
| `Pre_Contact` | Continuous (1–7) | Intergroup contact quality before intervention |
| `Post_Contact` | Continuous (1–7) | Intergroup contact quality after intervention |
| `Group_Identification` | Continuous (1–7) | Identification with own social group |
| `Intergroup_Anxiety` | Continuous (1–7) | Anxiety when interacting with outgroup |
| `Cooperation` | Continuous (1–7) | Cooperative behaviour during contact |
| `Life_Satisfaction` | Continuous (1–7) | General life satisfaction |

### Suggested Analyses

| Analysis | Module | What to test |
|---|---|---|
| Did prejudice change from pre to post? | **Compare Means** | Paired T-Test: `Pre_Prejudice` vs `Post_Prejudice` |
| Did the Contact group change more than Control? | **Analysis of Variance** | Between-Subjects ANOVA: DV = change score, Factor = `Condition` |
| Pre/post differences across both outcomes | **Analysis of Variance** | Repeated Measures: `Pre_Prejudice` + `Post_Prejudice` (check sphericity) |
| Does anxiety moderate contact → prejudice? | **Regression** | OLS with interaction: `Post_Prejudice` ~ `Post_Contact` × `Intergroup_Anxiety` |
| Does cooperation mediate contact → prejudice? | **Mediation Analysis** | X = `Post_Contact`, M = `Cooperation`, Y = `Post_Prejudice` |
| Full structural model | **SEM** | Contact → Cooperation → Prejudice, controlling for Anxiety |
| What predicts Life Satisfaction? | **Regression** | OLS: `Life_Satisfaction` ~ `Post_Contact` + `Cooperation` + `Intergroup_Anxiety` |
| Power for paired t-test replication | **Power Analysis** | A Priori, Paired T-Test, d = 0.30, α = .05, power = .80 |

---

## 2. `survey_example.csv` — Cross-Sectional Survey Study (N = 250)

### Research Context

This dataset simulates a cross-sectional survey assessing social identity,
intergroup contact, prejudice, and psychological wellbeing. Items were
measured on a 7-point Likert scale (1 = Strongly Disagree, 7 = Strongly Agree).
Four psychological constructs are measured using multi-item scales.

### Variable Codebook

| Variable | Type | Description |
|---|---|---|
| `ResponseID` | Integer | Participant identifier |
| `Age` | Integer | Age in years |
| `Gender` | Categorical | `Male`, `Female`, `Other` |
| `Education` | Categorical | `HS`, `Bachelor`, `Master`, `PhD` |
| `SI_1` – `SI_5` | Ordinal (1–7) | Social Identity scale items (5 items) |
| `IC_1` – `IC_4` | Ordinal (1–7) | Intergroup Contact scale items (4 items) |
| `Prejudice_1` – `Prejudice_3` | Ordinal (1–7) | Prejudice scale items (3 items) |
| `Wellbeing_1` – `Wellbeing_3` | Ordinal (1–7) | Wellbeing scale items (3 items) |

### Suggested Analyses

| Analysis | Module | What to test |
|---|---|---|
| Scale reliability & item quality | **Item Analysis & CVI** | Select SI_1–SI_5; check discrimination and α-if-deleted |
| Explore factor structure of all items | **Exploratory (EFA)** | Select all 15 items; promax rotation; check for 4 factors |
| Confirm the 4-factor model | **Confirmatory (CFA)** | Write lavaan syntax for SI, IC, Prejudice, Wellbeing factors |
| Correlations between scale totals | **Correlation** | Create composites in Data Management; Pearson correlation matrix |
| Does Social Identity predict Prejudice? | **Regression** | OLS: Prejudice composite ~ SI composite + IC composite |
| Does IC mediate SI → Prejudice? | **Mediation Analysis** | X = SI composite, M = IC composite, Y = Prejudice composite |
| Gender differences in Wellbeing | **Compare Means** | Independent T-Test: Wellbeing composite by `Gender` |
| Profile types based on all scales | **Cluster & Profile Analysis** | LPA with 3 classes using SI_1–SI_5 and Wellbeing_1–Wellbeing_3 |
| Item-level network of all scale items | **Network Analysis** | Select all 15 items; spring layout; check bridge items |
| Full SEM: SI → IC → Prejudice → Wellbeing | **SEM** | Build model in visual builder or write lavaan syntax |
| Dimension reduction of categorical items | **Categorical PCA (CATPCA)** | Select SI_1–SI_5 (auto-detected Ordinal); extract 2 dimensions |

### Quick Start: Compute Scale Composites

Before running regression or mediation on this dataset, create composite
scores in **Data Management → Compute & Transform**:

1. Select `SI_1`, `SI_2`, `SI_3`, `SI_4`, `SI_5` → click **Mean** → name it `SI_Total`
2. Select `IC_1`, `IC_2`, `IC_3`, `IC_4` → click **Mean** → name it `IC_Total`
3. Select `Prejudice_1`, `Prejudice_2`, `Prejudice_3` → click **Mean** → name it `Prejudice_Total`
4. Select `Wellbeing_1`, `Wellbeing_2`, `Wellbeing_3` → click **Mean** → name it `Wellbeing_Total`

These composites can then be used in any analysis module.

---

## Loading the Datasets in PsyStat

1. Open PsyStat and click **Data Management** in the navigation bar.
2. Click **📂 Load CSV/Excel/SPSS**.
3. Navigate to the `examples/` folder and select the dataset file.
4. The data will appear in the Dataset tab on the right.
5. You are ready to run analyses.
