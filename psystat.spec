# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for PsyStat (PyQt6) - Windows and macOS.

    pip install -r requirements-lock.txt pyinstaller
    pyinstaller --noconfirm psystat.spec

Output:  dist/PsyStat/          (Windows: folder with PsyStat.exe)
         dist/PsyStat.app       (macOS application bundle)

Verify a build with:  <built executable> --selftest selftest.log      (exit code 0 = all checks passed)

PsyStat imports most of its scientific libraries lazily, on first use, which PyInstaller cannot detect by itself.
Every such package is therefore listed in HIDDEN below. When you add a lazily imported package to psystat.py,
add it here too, and run the self-test.
"""
import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(SPECPATH)           # noqa: F821  (SPECPATH is provided by PyInstaller)
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")


def _version():
    with open(os.path.join(ROOT, "psystat.py"), encoding="utf-8") as fh:
        return re.search(r'APP_VERSION\s*=\s*"([^"]+)"', fh.read()).group(1)


VERSION = _version()


def _no_tests(name):
    return ".tests" not in name and ".testing" not in name and not name.endswith(".conftest")


# ── Hidden imports (lazily imported packages) ────────────────────────────────────────────────
HIDDEN = ["resources", "mistune", "openpyxl", "xlrd", "pyreadstat",
          "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_agg",
          "scipy.special._cdflib", "scipy.special.cython_special"]
for pkg in ("statsmodels", "pingouin", "semopy", "factor_analyzer", "prince", "altair", "seaborn",
            "pandas_flavor", "networkx.algorithms.community", "sklearn.covariance", "sklearn.mixture",
            "sklearn.cluster", "sklearn.decomposition", "sklearn.utils", "sklearn.neighbors"):
    try:
        HIDDEN += collect_submodules(pkg, filter=_no_tests)
    except Exception as exc:                                   # an optional package that is not installed
        print(f"[psystat.spec] skipping {pkg}: {exc}")

# ── Data files ───────────────────────────────────────────────────────────────────────────────
DATAS = []
for pkg in ("pingouin", "semopy", "factor_analyzer", "prince", "altair", "statsmodels"):
    try:
        DATAS += collect_data_files(pkg, excludes=["**/tests/**"])
    except Exception as exc:
        print(f"[psystat.spec] no data files for {pkg}: {exc}")

# PsyStat's own documents and example data (resource_path() finds them in the frozen app)
for rel, dest in [("README.md", "."), ("LICENSE", "."), ("NOTICE.md", "."), ("CITATION.cff", "."),
                  ("CHANGELOG.md", "."),
                  ("docs/USER_MANUAL.md", "docs"), ("docs/USER_MANUAL.html", "docs"), ("docs/USER_MANUAL.pdf", "docs"),
                  ("examples/EXAMPLES_GUIDE.md", "examples"), ("examples/experiment_example.csv", "examples"),
                  ("examples/survey_example.csv", "examples"), ("examples/cvi_expert_ratings_example.csv", "examples"),
                  ("examples/timeseries_example.csv", "examples")]:
    path = os.path.join(ROOT, rel)
    if os.path.exists(path):
        DATAS.append((path, dest))
    else:
        print(f"[psystat.spec] WARNING: {rel} not found - it will be missing from the build")

# ── Things that are definitely not needed (keeps the download small) ───────────────────────────────
EXCLUDES = [
    "tkinter", "_tkinter", "IPython", "jupyter", "notebook", "ipykernel", "pytest", "sphinx", "setuptools",
    "PyQt5", "PySide2", "PySide6", "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebChannel",
    "PyQt6.QtQml", "PyQt6.QtQuick", "PyQt6.QtQuickWidgets", "PyQt6.Qt3DCore", "PyQt6.QtMultimedia",
    "PyQt6.QtBluetooth", "PyQt6.QtSql", "PyQt6.QtNfc", "PyQt6.QtSensors", "PyQt6.QtSerialPort",
    "PyQt6.QtPositioning", "PyQt6.QtDesigner", "PyQt6.QtHelp", "PyQt6.QtTest",
    "matplotlib.backends.backend_tkagg", "matplotlib.backends.backend_tkcairo", "matplotlib.backends.backend_wx",
    "matplotlib.backends.backend_wxagg", "matplotlib.backends.backend_gtk3", "matplotlib.backends.backend_gtk4",
    "matplotlib.backends.backend_webagg", "matplotlib.backends.backend_nbagg",
    "torch", "tensorflow", "jax", "numba", "pymc", "arviz", "dask", "pyarrow", "cv2", "PIL.ImageTk",
]

ICON_WIN = os.path.join(ROOT, "assets", "icon.ico")
ICON_MAC = os.path.join(ROOT, "assets", "icon.icns")

a = Analysis(                                          # noqa: F821
    [os.path.join(ROOT, "psystat.py")],
    pathex=[ROOT],
    binaries=[],
    datas=DATAS,
    hiddenimports=sorted(set(HIDDEN)),
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)                                      # noqa: F821

exe = EXE(                                             # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PsyStat",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                    # UPX causes false virus-scanner alerts and occasional crashes
    console=False,                # windowed application (use --selftest <logfile> for diagnostics)
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=ICON_WIN if (IS_WIN and os.path.exists(ICON_WIN)) else None,
)
coll = COLLECT(                                        # noqa: F821
    exe, a.binaries, a.datas,
    strip=False, upx=False, name="PsyStat",
)

if IS_MAC:
    app = BUNDLE(                                      # noqa: F821
        coll,
        name="PsyStat.app",
        icon=ICON_MAC if os.path.exists(ICON_MAC) else None,
        bundle_identifier="io.github.terysetn02source.psystat",
        version=VERSION,
        info_plist={
            "CFBundleName": "PsyStat",
            "CFBundleDisplayName": "PsyStat",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Copyright (c) 2025-2026 Tery Setiawan. MIT License.",
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "Data file",
                "CFBundleTypeRole": "Viewer",
                "LSItemContentTypes": ["public.comma-separated-values-text"],
            }],
        },
    )
