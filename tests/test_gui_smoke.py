"""End-to-end smoke tests: drive the real interface headlessly with the example data and check key results."""
import re

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtWidgets import QTextEdit


def _text(tabs, before):
    out = []
    for i in range(before, tabs.count()):
        w = tabs.widget(i)
        edits = [w] if isinstance(w, QTextEdit) else w.findChildren(QTextEdit)
        out.append((tabs.tabText(i), " ".join(e.toPlainText() for e in edits)))
    return out


def _flat(tabs, before):
    names_text = _text(tabs, before)
    assert not [n for n, _ in names_text if "Error" in n], names_text
    return re.sub(r"\s+", " ", " ".join(t for _, t in names_text))


def _load(win, df):
    win.df = df
    win.populate_data_tables()
    win.update_global_dropdowns()


def _select(lst, names):
    lst.clearSelection()
    for i in range(lst.count()):
        if lst.item(i).text() in names:
            lst.item(i).setSelected(True)


def test_independent_t_test(window, examples):
    _load(window, examples["experiment"])
    window.t_dv.setCurrentText("contact_quality_post")
    window.t_iv.setCurrentText("condition")
    n = window.t_tabs.count()
    window.run_compare()
    t = _flat(window.t_tabs, n)
    assert "t = 10.932" in t and "1.269" in t


def test_cfa_ml_mlr_raw_and_fiml(window, examples):
    s = examples["survey"].copy()
    s["si5"] = 6 - s["si5"]
    _load(window, s)
    window.cfa_syntax.setPlainText("SI =~ si1+si2+si3+si4+si5\nIC =~ ic1+ic2+ic3+ic4\nPR =~ pr1+pr2+pr3\nWB =~ wb1+wb2+wb3")
    for est, metric, miss, expect in [(0, 0, 0, "Estimator: ML"), (1, 1, 0, "Estimator: MLR"), (0, 0, 1, "Missing data: FIML")]:
        window.cfa_estimator_combo.setCurrentIndex(est)
        window.cfa_metric_combo.setCurrentIndex(metric)
        window.cfa_missing_combo.setCurrentIndex(miss)
        n = window.cfa_tabs.count()
        window.run_cfa()
        t = _flat(window.cfa_tabs, n)
        assert expect in t and "Latent Variables" in t
        if miss == 0:
            assert "CFI" in t
        else:
            assert "not reported for FIML" in t


def test_network_ebicglasso_and_communities(window, examples):
    s = examples["survey"].copy()
    s["si5"] = 6 - s["si5"]
    _load(window, s)
    items = [c for c in s.columns if c[:2] in ("si", "ic", "pr", "wb") and c[2:].isdigit()]
    _select(window.sna_vars, items)
    window.sna_corr_type.setCurrentIndex(1)                     # Spearman
    n = window.sna_tabs.count()
    window.run_sna()
    t = _flat(window.sna_tabs, n)
    assert "EBICglasso" in t and "Communities detected: 4" in t
    plot = [i for i in range(n, window.sna_tabs.count()) if window.sna_tabs.tabText(i).startswith("Network Plot")]
    assert plot and hasattr(window.sna_tabs.widget(plot[0]), "canvas")


def test_mediation_reports_r_squared(window, examples):
    _load(window, examples["experiment"])
    _select(window.med_ivs, ["condition_code"])
    window.med_m.setCurrentText("contact_quality_post")
    window.med_y.setCurrentText("prejudice_post")
    window.med_n_boot.setValue(300)
    n = window.med_tabs.count()
    window.run_mediation()
    t = _flat(window.med_tabs, n)
    assert "Variance Explained (R²)" in t and "ΔR²" in t


def test_bayesian_independent_t_reports_bf01_and_prior(window, examples):
    _load(window, examples["experiment"])
    for lst, names in ((window.bayes_dv, ["age"]), (window.bayes_iv, ["condition"])):
        lst.clear()
        for c in examples["experiment"].columns:
            lst.addItem(c)
        _select(lst, names)
    idx = [i for i in range(window.bayes_type.count()) if "Independent" in window.bayes_type.itemText(i)][0]
    window.bayes_type.setCurrentIndex(idx)
    n = window.bayes_tabs.count()
    window.run_bayes()
    t = _flat(window.bayes_tabs, n)
    assert "BF01" in t and "Cauchy(0, 0.707)" in t and "95% CrI" in t and "for reference" in t


def test_repeated_measures_has_sphericity(window, examples):
    e = examples["experiment"]
    _load(window, e[e.condition == "Contact"].reset_index(drop=True))
    window.anova_design_rm.setChecked(True)
    _select(window.rm_vars, ["prejudice_pre", "prejudice_post", "prejudice_followup"])
    n = window.anova_tabs.count()
    window.run_repeated_measures()
    t = _flat(window.anova_tabs, n)
    assert "Mauchly" in t and "Greenhouse" in t


def test_catpca_numeric_items(window, examples):
    pytest.importorskip("prince")
    s = examples["survey"]
    _load(window, s)
    items = [c for c in s.columns if c[:2] in ("si", "ic", "pr", "wb") and c[2:].isdigit()]
    _select(window.catpca_vars, items)
    window.catpca_k.setValue(3)
    n = window.catpca_tabs.count()
    window.run_catpca()
    _flat(window.catpca_tabs, n)                                # no error tab
