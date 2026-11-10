"""Core numerical routines are checked against independent implementations."""
import numpy as np
import pandas as pd
import pytest


def test_bayes_correlation_matches_pingouin(ps):
    pg = pytest.importorskip("pingouin")
    for r, n, kappa in [(0.3, 50, 1.0), (0.1, 100, 1.0), (-0.45, 30, 1.0), (0.3, 50, 0.5)]:
        mine = ps._bayes_corr(r, n, kappa)["bf10"]
        assert mine == pytest.approx(pg.bayesfactor_pearson(r, n, kappa=kappa), rel=1e-4)


def test_bayes_ttest_matches_pingouin(ps):
    pg = pytest.importorskip("pingouin")
    for t, n1, n2, r in [(2.5, 30, 30, 0.707), (0.4, 25, 40, 0.707), (2.5, 30, 30, 1.0)]:
        mine = ps._bayes_t(t, n1 * n2 / (n1 + n2), n1 + n2 - 2, r)["bf10"]
        assert mine == pytest.approx(pg.bayesfactor_ttest(t, n1, n2, r=r), rel=1e-4)


def test_bayes_prior_changes_result(ps):
    assert ps._bayes_t(2.0, 15, 58, 0.707)["bf10"] != pytest.approx(ps._bayes_t(2.0, 15, 58, 1.0)["bf10"], rel=1e-3)


def test_p_adjustment_matches_statsmodels(ps):
    mt = pytest.importorskip("statsmodels.stats.multitest")
    p = np.random.default_rng(0).uniform(0, 0.2, 40) ** 1.5
    for how, name in ((0, "bonferroni"), (1, "holm"), (2, "fdr_bh")):
        assert np.allclose(ps._sna_adjust_p(p, how), mt.multipletests(p, method=name)[1])


def test_ebicglasso_recovers_chain(ps):
    rng = np.random.default_rng(11)
    p = 8
    K = np.eye(p)
    for i in range(p - 1):
        K[i, i + 1] = K[i + 1, i] = -0.3
    X = rng.multivariate_normal(np.zeros(p), np.linalg.inv(K), size=600)
    W, info = ps._sna_estimate(X, 0, 0)
    iu = np.triu_indices(p, 1)
    true_edge = np.abs(np.subtract.outer(np.arange(p), np.arange(p)))[iu] == 1
    assert (W[iu][true_edge] != 0).all()                       # all true edges found
    assert (W[iu][true_edge] > 0).all()                        # with the right sign


def test_singular_correlation_is_detected(ps):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    X = np.column_stack([X, X.sum(axis=1)])                     # a "total score" with its items
    with pytest.raises(np.linalg.LinAlgError):
        ps._sna_estimate(X, 1, 0)


def test_cfa_helpers_agree_with_semopy(ps):
    semopy = pytest.importorskip("semopy")
    rng = np.random.default_rng(7)
    n = 400
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    d = {f"a{i}": f1 * l + rng.normal(size=n) * .6 for i, l in enumerate([.8, .7, .6], 1)}
    d.update({f"b{i}": f2 * l + rng.normal(size=n) * .6 for i, l in enumerate([.8, .7, .6], 1)})
    df = pd.DataFrame(d)
    m = semopy.Model("A =~ a1+a2+a3\nB =~ b1+b2+b3")
    m.fit(df, tol=1e-10)
    ins = m.inspect()
    obs = list(m.vars["observed"])
    st = semopy.calc_stats(m)
    res = ps._cfa_robust(ins, obs, df[obs].values, float(st["chi2"].values[0]), float(st["chi2 Baseline"].values[0]))
    se_sem = pd.to_numeric(ins["Std. Err"], errors="coerce").values
    ok = np.isfinite(se_sem)
    assert np.allclose(res["se_ml_check"][ok], se_sem[ok], rtol=1e-3)         # classical SE reproduced
    assert 0.5 < res["c"] < 1.5 and res["CFI"] <= 1.0
    sa, _ = ps._cfa_standardize(ins, obs, set(m.vars["latent"]))
    assert np.nanmax(sa["Est. Std"].values) <= 1.0 + 1e-9


def test_p_value_format_has_no_leading_zero(ps, window):
    assert window.apa_p(0.032) == "p = .032"
    assert window.apa_p(0.0004).startswith("p &lt; .001")


def test_cell_editing_rejects_text_in_numeric_column(ps, window, monkeypatch):
    shown = []
    monkeypatch.setattr(ps.QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[2])))
    window.df = pd.DataFrame({"x": [1.5, 2.5, 3.5]})
    window.populate_data_tables()
    item = window.data_table.item(1, 0)
    item.setText("5.a2")                                                # not a number
    assert len(shown) == 1 and "not a valid number" in shown[0]         # the user is told
    assert window.df["x"].dtype.kind == "f" and window.df.iloc[1, 0] == 2.5     # column intact, old value restored
    assert window.data_table.item(1, 0).text() == "2.5"
    item.setText("7,25")                                                # decimal comma accepted
    assert window.df.iloc[1, 0] == pytest.approx(7.25)


def test_text_allowed_in_new_blank_column(ps, window):
    window.df = pd.DataFrame({"x": [1.0, 2.0], "label": [float("nan"), float("nan")]})
    window.populate_data_tables()
    window.data_table.item(0, 1).setText("M")
    assert window.df.iloc[0, 1] == "M"
