"""The example datasets contain the effects described in examples/EXAMPLES_GUIDE.md; resources are found."""
import numpy as np
import pytest
from scipy import stats


def test_shapes(examples):
    assert examples["experiment"].shape == (300, 18)
    assert examples["survey"].shape == (250, 19)
    assert examples["cvi"].shape == (12, 7)
    assert examples["timeseries"].shape == (60, 2)


def test_experiment_contact_effect(examples):
    e = examples["experiment"]
    a = e.loc[e.condition == "Contact", "contact_quality_post"].dropna()
    b = e.loc[e.condition == "Control", "contact_quality_post"].dropna()
    t, p = stats.ttest_ind(a, b)
    assert len(a) == 147 and len(b) == 150
    assert t == pytest.approx(10.93, abs=0.02) and p < 1e-20
    assert a.mean() == pytest.approx(5.06, abs=0.01) and b.mean() == pytest.approx(3.93, abs=0.01)


def test_experiment_missingness(examples):
    e = examples["experiment"]
    assert e["prejudice_followup"].isna().mean() == pytest.approx(0.08, abs=0.04)
    assert e["participant_id"].notna().all() and e["condition"].notna().all()


def test_survey_reverse_item_and_missing(examples):
    s = examples["survey"].copy()
    items = [c for c in s.columns if c[:2] in ("si", "ic", "pr", "wb") and c[2:].isdigit()]
    assert len(items) == 15
    assert int(s[items].dropna().shape[0]) == 207                      # listwise N used in the guide
    si = ["si1", "si2", "si3", "si4", "si5"]

    def alpha(df):
        df = df.dropna()
        k = df.shape[1]
        return k / (k - 1) * (1 - df.var().sum() / df.sum(axis=1).var())
    raw = alpha(s[si])
    s["si5"] = 6 - s["si5"]
    assert raw < 0.5 < alpha(s[si]) and alpha(s[si]) == pytest.approx(0.75, abs=0.03)


def test_cvi_example(examples):
    r = examples["cvi"].filter(like="expert")
    icvi = (r > 2.5).mean(axis=1)
    assert icvi.mean() == pytest.approx(0.83, abs=0.01)
    assert (icvi == 1).mean() == pytest.approx(0.67, abs=0.01)


def test_resources_resolve(ps):
    import resources
    for rel in ("README.md", "LICENSE", "docs/USER_MANUAL.md", "docs/USER_MANUAL.html", "examples/EXAMPLES_GUIDE.md"):
        assert resources.resource_path(rel)
    info = resources.get_example_dataset_info()
    assert len(info) == 4
    for fname in info:
        assert len(resources.load_example_dataset(fname)) >= 10
    with pytest.raises(FileNotFoundError):
        resources.resource_path("no/such/file.txt")


def test_markdown_renders(ps):
    import resources
    html = resources.render_markdown("# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", "t")
    assert "<h1>" in html and "<table>" in html
