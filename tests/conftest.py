"""Shared test fixtures. The application is imported as a module and driven headlessly (no screen required)."""
import importlib.util
import os
import sys
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def ps():
    """The psystat module (imported once)."""
    spec = importlib.util.spec_from_file_location("psystat_app", os.path.join(ROOT, "psystat.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def qapp(ps):
    return ps.QApplication.instance() or ps.QApplication(sys.argv[:1])


@pytest.fixture()
def window(ps, qapp, monkeypatch, tmp_path):
    """A fresh main window. Dialogs never block: warnings raise, so a test cannot hang on a modal box."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))          # isolated autosave folder -> no recovery prompt

    def _no_dialog(*args, **kwargs):
        raise AssertionError("unexpected dialog: " + " | ".join(str(a) for a in args[1:3]))
    for name in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(ps.QMessageBox, name, staticmethod(_no_dialog))
    win = ps.PsyStat()
    yield win
    win.close()


@pytest.fixture(scope="session")
def examples():
    import pandas as pd
    d = os.path.join(ROOT, "examples")
    return {
        "experiment": pd.read_csv(os.path.join(d, "experiment_example.csv")),
        "survey": pd.read_csv(os.path.join(d, "survey_example.csv")),
        "cvi": pd.read_csv(os.path.join(d, "cvi_expert_ratings_example.csv")),
        "timeseries": pd.read_csv(os.path.join(d, "timeseries_example.csv")),
    }
