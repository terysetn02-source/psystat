# psystat.spec
# ------------
# PyInstaller build specification for PsyStat.
#
# Usage:
#   pip install pyinstaller
#   pyinstaller psystat.spec
#
# Output:
#   dist/PsyStat          (macOS .app or Windows folder)
#   dist/PsyStat.exe      (Windows one-file build, if onefile=True)
#
# The ISS installer script (installer/psystat_setup.iss) wraps the
# dist/ output into the final user-facing Setup.exe for Windows.

import sys
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE

# ── Bundled data files ────────────────────────────────────────────────────────
# Each tuple: (source_path, destination_folder_inside_bundle)
datas = [
    ("docs/USER_MANUAL.md",              "docs"),
    ("examples/experiment_example.csv",  "examples"),
    ("examples/survey_example.csv",      "examples"),
    ("examples/EXAMPLES_GUIDE.md",       "examples"),
    ("resources.py",                     "."),
]

# ── Hidden imports ────────────────────────────────────────────────────────────
# Packages that PyInstaller's static analyser misses (dynamic imports,
# optional imports, etc.)
hiddenimports = [
    # Core scientific stack
    "numpy", "pandas", "scipy", "scipy.stats", "scipy.optimize",
    # statsmodels — many submodules are dynamically imported
    "statsmodels", "statsmodels.api", "statsmodels.formula.api",
    "statsmodels.stats.anova", "statsmodels.stats.multitest",
    "statsmodels.stats.mediation",
    # Psychometrics
    "factor_analyzer", "factor_analyzer.factor_analyzer",
    "semopy", "semopy.model",
    "pingouin",
    "prince",
    # Clustering / ML
    "sklearn", "sklearn.mixture", "sklearn.cluster",
    "sklearn.preprocessing", "sklearn.decomposition",
    # Network
    "networkx",
    # Plotting
    "matplotlib", "matplotlib.backends.backend_qt6agg",
	"PyQt6", "PyQt6.QtWidgets", "PyQt6.QtCore", "PyQt6.QtGui",
	"PyQt6.QtWebEngineWidgets",
    # Markdown renderer (optional; resources.py has a fallback)
    "mistune",
    # File I/O
    "openpyxl", "xlrd", "pyreadstat",
]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["psystat.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "test", "unittest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

# ── EXE (Windows) / Unix executable ──────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PsyStat",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window on launch
    icon="assets/icon.ico" if sys.platform == "win32" else "assets/icon.icns",
)

# ── COLLECT — folder-based build (used by ISS installer on Windows) ───────────
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PsyStat",
)

# ── BUNDLE — macOS .app ───────────────────────────────────────────────────────
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PsyStat.app",
        icon="assets/icon.icns",
        bundle_identifier="id.ac.maranatha.psystat",
        info_plist={
            "CFBundleDisplayName":        "PsyStat",
            "CFBundleShortVersionString": "1.1.0",
            "CFBundleVersion":            "1.1.0",
            "NSHumanReadableCopyright":   "Copyright © 2025 Tery Setiawan. MIT License.",
            "NSHighResolutionCapable":    True,
        },
    )
