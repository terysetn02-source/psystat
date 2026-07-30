import sys
import os
import unittest  # Add this to force the bundler to include it

# Built-in resource loader — resolves paths to bundled docs and datasets
# whether running from source or as a frozen PyInstaller / Nuitka bundle.
try:
    from resources import resource_path
except ImportError:
    def resource_path(relative_path: str) -> str:
        """Fallback: resolve relative to this file's directory."""
        base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, relative_path.replace("/", os.sep))
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM  # retained for fallback; primary RM-ANOVA now uses pingouin
try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False

try:
    import prince
    PRINCE_AVAILABLE = True
except ImportError:
    PRINCE_AVAILABLE = False
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.miscmodels.ordinal_model import OrderedModel
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import warnings
from datetime import datetime

# --- CRITICAL FIX: Sklearn Incompatibility Patch ---
warnings.filterwarnings('ignore')
try:
    import sklearn.utils.validation as skval
    import inspect
    _original_check_array = skval.check_array
    _check_array_sig = inspect.signature(_original_check_array)
    
    def _patched_check_array(*args, **kwargs):
        if 'force_all_finite' in kwargs and 'ensure_all_finite' in _check_array_sig.parameters:
            kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
        elif 'ensure_all_finite' in kwargs and 'force_all_finite' in _check_array_sig.parameters:
            kwargs['force_all_finite'] = kwargs.pop('ensure_all_finite')
        return _original_check_array(*args, **kwargs)
        
    skval.check_array = _patched_check_array
except ImportError:
    pass

# --- Scientific Dependencies ---
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    import factor_analyzer.factor_analyzer as fafa
    import factor_analyzer.utils as fautils
    if hasattr(fafa, 'check_array'): fafa.check_array = _patched_check_array
    if hasattr(fautils, 'check_array'): fautils.check_array = _patched_check_array
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False

try:
    from semopy import Model, ModelMeans, calc_stats
    SEM_AVAILABLE = True
except ImportError:
    SEM_AVAILABLE = False

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from statsmodels.stats.power import TTestIndPower, TTestPower, FTestAnovaPower, NormalIndPower, GofChisquarePower

# PsyStat app icon embedded as base64 PNG -- a statistics-themed emblem (normal
# distribution curve with a centre line) rendered on a dark professional background.
# Embedded directly so the app has no external icon-file dependency.
PSYSTAT_ICON_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAIJElEQVR4nOVbS28b1xX+OENSEmck"
    "e0RJlEhRIllZLwtSRLNVqjhKbMOJE6dJACVAgKCtUXfToj+haLtol+2uKLoIAqRdFEW6qDdG0KSO"
    "WztGHDmR5FiKLCl6i6QoipT5fs10EZPi6A4pcV4ukA8QIH733sMz39x77sy5h4augWcEfItBPWkH"
    "njS+9QIYn8SXUpQJPT0X4XB4IQg8/P5pLC7+Czyf090Xg94xwGKx4oUXfocOqw1W40KJXwyZce36"
    "b5FK7enpjr4C0LQZr776RzQ3e9Bh/AxvNL5Rans/9j7mQi24du0XyOfTermkbwzw+X6K5mZPxXaO"
    "c8Pnu6qjRzoK0NTUicHB14/sNzDwGpqaHNo79Bi6CTA6+kNQFC0mhcd/5Q5RNEZHf6SXW/oIYLFY"
    "4fGcI/hCIY9CIU/wHs85NDQ06+GaPgL09l4i7n5sPwRBKDz+E08DiqLR23tJD9f0EMCAvr7LBLu5"
    "fr/0P18oEO39/a8AMGjpGAAdBLDZToNlbSIuk0kiFFwtfRYEHrmseOtjWRtsttNau6e9AG73cwS3"
    "sToDgS+/6wKC/sVjjVUbGgtggMs1QbCb618SXCjwNcG5XBMwGLR1UVPrNttpMEyLiEsl97G3u0n0"
    "fbS/g2QiKuIYpkXzZaCpAN3dZwlua2MOxOZfantAcE7n99V2SwRNBejs/C7Bba2TF1nE9sY8wTmd"
    "31PVp8PQTACGaQPHuURcNpNEOLxRccxeeBPZTFLEcZwbLNumhYsANBSgs9NHcDuBZUCo/PIpCDyC"
    "gSWCdzqfVtW3cmgmgMNBTv+gn7y4wwhsPyS4zs4xVXySgiYCGAw0HA4vwUvdXaKPfwmCwIs4u30U"
    "NG1Wzb9yaCJAW9sAzGZWxEUjAaRTsSPHZjNJ7IW3RJzRWIf29hFVfSxCEwGkor/Uk14lBCWWgVa7"
    "gSYCOBxSAfDo6V9EQEIsqZiiBlQXoL7+JFpb+0VcPpdBOLR+bBvRPT/S6biIO3nSicbGDlV8LIfq"
    "AjgcZwguFFwBz5OvvJUhYEdix5BaWkqhugBK1//BGFIALZaBygIYJNd/4Bj7/2EEA9LbIUWZZHsn"
    "BVUFsFp70NDAAQBcLgouF4V4LIxkIlKzrWwmicjedskOAJhMDbDZhtR0WV0BpKd/7Xe/2li1t0OV"
    "BSCdk7P+q41VOxCqJoDZzKCtbVDE8TyPUHBFts1IeAv5vPjAlOPcYJhW2TYPQzUB7PYzROo7EY+g"
    "UJB/4isIPOKxMMGrOQtUE+CwU9GogIV5+dO/iK/mFhCNil+h1dwONRXgi6lPFdv9/O4dCQG8MBjo"
    "CiNqgyoCcJyLWJfJRBSxRyHFttPpOKIRv4gzm1nYbIMVRtQGVQSQyticPLGP8WfJh6JaMf6sD81c"
    "guAdDnW2Q1UEkMr+5rJRMIxFsW2GsSCf2yd4p1OdOKBYAIulBW1tAyIun89KRm+5SCaiyOXER2dW"
    "6ylVTpAVC9DdPU5wO4Fl8Dwv0VseBEGQPDnq6lJ+ZqCCAOT0394k8/tKIZUk8XieV2xXkQB1dU3o"
    "6BDn6ni+AP/WQoUR8rG9MU/kFNrbR1Bfzymyq6hO0ON5HhQlNrG7s4ZcNoXZ6TlFjhVRtJPNJhEK"
    "rsDW0VNqoygabvcE5uf/Kdu+ohnQ03OR4IqFD8HALoKBXSXmCTvlRRVFKF0GsgVobLQTLz+FQr7q"
    "2Z9SbG/Mgz9UU9TePqyoqky2AL29LxJcYGuhtF1dvDSBi5fI2oBaUW4nl0tL5gj6+l6WbV+WADRt"
    "kqz7WV+dke3IcbGyfI/gTp16kYhFx4UsAdzu50qpryLSqZjkVqU2gv5FpJLiJ8OGBk52OY0MAQwY"
    "GnqTYL9e+uxQ3Y82EAQeq8ufE/zw8FuQU1VWswDd3c/Aau0RcXwhj5WlqZq/XC5WlqeIYNjc7EFX"
    "V+3H6DUJYDBQ8Hp/TPAba7PIHDrJ0RLpVAxrK9ME7/NdrbmoqqbIMTj4OlHtzfMFfPXgJtH3k1vq"
    "zIhKdh7O34LL44WBOrhgjnOjv/8HNT0YHVsulm2H13uF4NdXZ5CIk3n/RDyJRDxJ8LWikp1EfA9r"
    "q9ME7/NdrekM8VgC0LQJFy78GmYzI+JzuTTmZj+SHMOwFjCsCvmAKnYezHyIXC4j4sxmBufO/RI0"
    "XXcs+0cKQFEmXLjwG7S09BJtc7MfVSx6GD/rw/hZFTJCVexk0nHM379B8K2t/Th//leg6aOP0aoK"
    "wLI2XL78B8mUV2hnFcuLd4/8Aq2x9PAOdoJSuYKn8dJLvwfLtlcdLykAx7kxNvYzTE6+SzzvA99E"
    "4bu3/1614ks3CAKm7vxDcibabKcxOfkOxsZ+Do5zSw4v7QI0bcLw8Fvo63uFKG8tRzabwic3/6rr"
    "tncU0qkYbt/8CybO/wQmc72ozWisx9DQJIaGJhGPB7GwcB2zs38r/USvNANGRt6G13ul6sVn0gnc"
    "vvEekab+f8B+JIBbH79HVJaUg2VtOHPmCp566u0SVxJAqrKjHOHQOv79wZ8Q2duq2u9JIhLexI0P"
    "/ozd0FrVfnb7QQlfaQlEIquS6z2ZeITFuXvYXF0AwKLOyBJ9pPCfD795ba0z2iXbTcY8ykOQydgi"
    "2fcoO4fBZ4FPP74Op6sf3xkYBcOeIPpEIgcHtiUBpqbeQX39CXR0jCKTiSHgn8Pm+hS2t2eI3/So"
    "gVg6AkHIl32eRiRZ/c7Vgsjcf3F/3gC7fQSdXT7YHcMwmSwIBGZw7967pX66/3S2CMZcwKmWg1z/"
    "4m49Ell1zvtqwf8AcanXGmHv0dwAAAAASUVORK5CYII="
)

def _make_psystat_qicon():
    """Decodes the embedded base64 PNG icon and returns a QIcon for the main window."""
    try:
        import base64
        raw = base64.b64decode(PSYSTAT_ICON_B64)
        pix = QPixmap()
        pix.loadFromData(raw)
        return QIcon(pix)
    except Exception:
        return QIcon()

def format_nav_label(label):
    """
    Smart two-line formatter for nav bar button labels.
    Each label is split into at most two balanced lines so the text fits
    comfortably inside the nav button without cropping or overflow.
    Hard-coded overrides are used wherever an automatic split would look odd.
    """
    overrides = {
        "Data Management":                   "Data\nManagement",
        "Data Visualization":                "Data\nVisualization",
        "Descriptives & Crosstabs":          "Descriptives\n& Crosstabs",
        "Item Analysis & CVI":               "Item Analysis\n& CVI",
        "Correlation":                       "Correlation",
        "Compare Means":                     "Compare\nMeans",
        "Analysis of Variance":              "Analysis of\nVariance",
        "Regression":                        "Regression",
        "Mediation Analysis":                "Mediation\nAnalysis",
        "Categorical PCA (CATPCA)":          "Categorical\nPCA (CATPCA)",
        "Cluster & Profile Analysis (LCA)":  "Cluster & Profile\nAnalysis (LCA)",
        "Forecasting (Time Series)":         "Forecasting\n(Time Series)",
        "Exploratory (EFA)":                 "Exploratory\n(EFA)",
        "Confirmatory (CFA)":                "Confirmatory\n(CFA)",
        "SEM (Graph & Syntax)":              "SEM\n(Graph & Syntax)",
        "Network Analysis":                  "Network\nAnalysis",
        "Power Analysis":                    "Power\nAnalysis",
    }
    if label in overrides:
        return overrides[label]
    # Fallback: split near the midpoint on a word boundary
    if len(label) <= 12:
        return label
    words = label.split()
    mid_char = len(label) // 2
    best_split, best_dist, pos = 1, float('inf'), 0
    for i, w in enumerate(words[:-1]):
        pos += len(w) + 1
        dist = abs(pos - mid_char)
        if dist < best_dist:
            best_dist = dist
            best_split = i + 1
    return ' '.join(words[:best_split]) + '\n' + ' '.join(words[best_split:])


def generate_nav_icon(kind, dark=False, size_px=48, dpi=100):
    """
    Renders a tiny matplotlib thumbnail that visually hints at what an analysis
    module does (e.g. a bell curve for Descriptives, a scatter cloud for
    Correlation), returned as a QPixmap. Used by the top navigation bar instead
    of plain text labels. Rendered once at startup and cached by the caller --
    these are not meant to be regenerated on every click.
    Returns None if matplotlib isn't available, so callers can fall back to a
    text-only button.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    accent = '#818CF8' if dark else '#4F46E5'
    accent2 = '#34D399' if dark else '#10B981'
    accent3 = '#FBBF24' if dark else '#F59E0B'
    fig = Figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    rng = np.random.default_rng(7)

    try:
        if kind == "Data Management":
            # Small spreadsheet grid
            for i in range(4):
                ax.axhline(i / 3, color=accent, linewidth=0.8, alpha=0.6)
                ax.axvline(i / 3, color=accent, linewidth=0.8, alpha=0.6)
        elif kind == "Data Visualization":
            x = np.linspace(0, 1, 30)
            ax.plot(x, 0.5 + 0.3 * np.sin(x * 6), color=accent, linewidth=1.8)
            ax.fill_between(x, 0, 0.5 + 0.3 * np.sin(x * 6), color=accent, alpha=0.15)
        elif "Descriptives" in kind:
            x = np.linspace(-3, 3, 100)
            y = np.exp(-x**2 / 2)
            ax.plot(x, y, color=accent, linewidth=1.8)
            ax.fill_between(x, 0, y, color=accent, alpha=0.2)
        elif "Item Analysis" in kind:
            vals = [0.8, 0.6, 0.9, 0.7]
            ax.bar(range(4), vals, color=accent, width=0.6)
            ax.axhline(0.7, color=accent3, linewidth=1, linestyle='--')
        elif kind == "Correlation":
            x = rng.normal(0, 1, 25)
            y = 0.7 * x + rng.normal(0, 0.5, 25)
            ax.scatter(x, y, color=accent, s=10, alpha=0.7)
            ax.plot([-2, 2], [-1.4, 1.4], color=accent3, linewidth=1.2)
        elif "Compare Means" in kind:
            ax.bar([0, 1], [0.4, 0.7], color=[accent, accent2], width=0.5)
            ax.plot([0, 1], [0.55, 0.85], color=accent3, linewidth=1)
        elif "Analysis of Variance" in kind:
            ax.bar([0, 1, 2], [0.4, 0.75, 0.55], color=[accent, accent2, accent3], width=0.6)
        elif kind == "Regression":
            x = rng.normal(0, 1, 25)
            y = 0.6 * x + rng.normal(0, 0.4, 25)
            ax.scatter(x, y, color=accent, s=8, alpha=0.6)
            ax.plot([-2, 2], [-1.2, 1.2], color=accent3, linewidth=1.5)
        elif "Mediation" in kind:
            pts = {'X': (0.05, 0.1), 'M': (0.5, 0.85), 'Y': (0.95, 0.1)}
            for (x1, y1), (x2, y2) in [(pts['X'], pts['M']), (pts['M'], pts['Y']), (pts['X'], pts['Y'])]:
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color=accent, lw=1.2))
            for name, (x, y) in pts.items():
                ax.scatter([x], [y], color=accent3, s=60, zorder=5)
        elif "CATPCA" in kind:
            theta = np.linspace(0, 2 * np.pi, 8)
            ax.scatter(np.cos(theta) * 0.4 + 0.5, np.sin(theta) * 0.4 + 0.5, color=accent, s=15)
            ax.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5), arrowprops=dict(arrowstyle='->', color=accent3, lw=1.2))
        elif "Cluster" in kind or "LCA" in kind:
            c1 = rng.normal([0.3, 0.3], 0.08, (8, 2))
            c2 = rng.normal([0.7, 0.7], 0.08, (8, 2))
            ax.scatter(c1[:, 0], c1[:, 1], color=accent, s=12)
            ax.scatter(c2[:, 0], c2[:, 1], color=accent2, s=12)
        elif "Forecasting" in kind:
            x = np.linspace(0, 1, 20)
            y = 0.3 + 0.3 * x + 0.05 * np.sin(x * 20)
            ax.plot(x, y, color=accent, linewidth=1.5)
            xf = np.linspace(1, 1.3, 8)
            yf = 0.3 + 0.3 * xf
            ax.plot(xf, yf, color=accent3, linewidth=1.5, linestyle='--')
        elif "EFA" in kind:
            ax.bar(range(5), [0.9, 0.6, 0.35, 0.2, 0.1], color=accent, width=0.6)
        elif "CFA" in kind:
            pts = {'F': (0.5, 0.85), 'i1': (0.15, 0.1), 'i2': (0.5, 0.1), 'i3': (0.85, 0.1)}
            for k2 in ('i1', 'i2', 'i3'):
                ax.annotate('', xy=pts[k2], xytext=pts['F'], arrowprops=dict(arrowstyle='->', color=accent, lw=1.1))
            ax.scatter(*pts['F'], color=accent3, s=70, zorder=5)
            for k2 in ('i1', 'i2', 'i3'):
                ax.scatter(*pts[k2], color=accent2, s=35, zorder=5, marker='s')
        elif "SEM" in kind:
            pts = {'A': (0.1, 0.7), 'B': (0.5, 0.7), 'C': (0.9, 0.3)}
            ax.annotate('', xy=pts['B'], xytext=pts['A'], arrowprops=dict(arrowstyle='->', color=accent, lw=1.2))
            ax.annotate('', xy=pts['C'], xytext=pts['B'], arrowprops=dict(arrowstyle='->', color=accent, lw=1.2))
            for _, (x, y) in pts.items():
                ax.scatter([x], [y], color=accent3, s=50, zorder=5)
        elif "Network" in kind:
            theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            xs, ys = np.cos(theta) * 0.35 + 0.5, np.sin(theta) * 0.35 + 0.5
            for i in range(6):
                for j in range(i + 1, 6):
                    if rng.random() > 0.55:
                        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=accent, linewidth=0.7, alpha=0.6)
            ax.scatter(xs, ys, color=accent3, s=30, zorder=5)
        elif "Power" in kind:
            x = np.linspace(0, 1, 40)
            y = 1 / (1 + np.exp(-12 * (x - 0.5)))
            ax.plot(x, y, color=accent, linewidth=1.8)
            ax.axhline(0.8, color=accent3, linewidth=1, linestyle='--')
        else:
            ax.scatter([0.5], [0.5], color=accent, s=40)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        fig.tight_layout(pad=0.15)

        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, dpi=dpi)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        return pixmap
    except Exception:
        return None

def calc_cronbach_alpha(df):
    k = df.shape[1]
    if k < 2: return np.nan
    var_items = df.var(ddof=1).sum()
    var_total = df.sum(axis=1).var(ddof=1)
    if var_total == 0 or pd.isna(var_total): return np.nan
    return (k / (k - 1)) * (1 - var_items / var_total)

def calc_mcdonalds_omega(df):
    """
    McDonald's Omega (total), computed from a single-factor model fit to the items.
    omega = (sum of standardized loadings)^2 / [(sum of loadings)^2 + sum of (1 - loadings^2)]
    This is the standard formula (McDonald, 1999; Hayes & Coutts, 2020), using a
    single common factor extracted from the item covariance matrix. Unlike alpha,
    omega does not assume tau-equivalence (equal loadings), so it tends to be a
    less biased reliability estimate when item loadings differ.
    Returns (omega, loadings_series) or (np.nan, None) if it cannot be estimated.
    """
    k = df.shape[1]
    if k < 3 or not FA_AVAILABLE:
        return np.nan, None
    try:
        clean = df.dropna()
        if clean.shape[0] < k + 2:
            return np.nan, None
        # Drop zero-variance columns, which break factor extraction
        clean = clean.loc[:, clean.std(ddof=1) > 0]
        if clean.shape[1] < 3:
            return np.nan, None
        fa = FactorAnalyzer(n_factors=1, rotation=None, method='minres')
        fa.fit(clean)
        loadings = fa.loadings_[:, 0]
        loadings = pd.Series(loadings, index=clean.columns)
        sum_load = loadings.sum()
        sum_err = (1 - loadings**2).sum()
        denom = (sum_load**2) + sum_err
        if denom <= 0:
            return np.nan, loadings
        omega = (sum_load**2) / denom
        return float(omega), loadings
    except Exception:
        return np.nan, None

class VarSelectDialog(QDialog):
    def __init__(self, available_vars, selected_vars, title="Select Variables", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(550, 450)
        
        layout = QHBoxLayout(self)
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Available Variables:"))
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.textChanged.connect(self.filter_available)
        left_layout.addWidget(self.search_bar)
        
        self.list_avail = QListWidget()
        self.list_avail.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        left_layout.addWidget(self.list_avail)
        layout.addLayout(left_layout)
        
        mid_layout = QVBoxLayout()
        mid_layout.addStretch()
        self.btn_add = QPushButton("►")
        self.btn_remove = QPushButton("◄")
        self.btn_add.clicked.connect(self.add_vars)
        self.btn_remove.clicked.connect(self.remove_vars)
        mid_layout.addWidget(self.btn_add)
        mid_layout.addWidget(self.btn_remove)
        mid_layout.addStretch()
        layout.addLayout(mid_layout)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Selected Variables:"))
        self.list_selected = QListWidget()
        self.list_selected.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        right_layout.addWidget(self.list_selected)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        right_layout.addWidget(btn_box)
        layout.addLayout(right_layout)
        
        self.all_avail = [v for v in available_vars if v not in selected_vars]
        self.list_avail.addItems(self.all_avail)
        self.list_selected.addItems(selected_vars)
        
    def filter_available(self, text):
        self.list_avail.clear()
        if text:
            filtered = [v for v in self.all_avail if text.lower() in v.lower()]
            self.list_avail.addItems(filtered)
        else:
            self.list_avail.addItems(self.all_avail)
            
    def add_vars(self):
        items = self.list_avail.selectedItems()
        for item in items:
            text = item.text()
            self.list_selected.addItem(text)
            self.all_avail.remove(text)
            self.list_avail.takeItem(self.list_avail.row(item))
            
    def remove_vars(self):
        items = self.list_selected.selectedItems()
        for item in items:
            text = item.text()
            self.all_avail.append(text)
            self.list_selected.takeItem(self.list_selected.row(item))
        self.filter_available(self.search_bar.text())
        
    def get_selected(self):
        return [self.list_selected.item(i).text() for i in range(self.list_selected.count())]

GLOBAL_WINDOWS = []

# ==========================================
# TOP NAVIGATION BAR (replaces the old left sidebar)
# ==========================================
class NavTopButton(QWidget):
    """
    A single item in the two-row top navigation bar. Built as a custom QWidget
    (not QToolButton) to avoid Qt's internal icon-vs-text sizing competition that
    caused icons to collapse to near-zero height when both an icon AND label text
    were requested in a constrained height. Instead we use an explicit VBox:
        - QLabel(icon pixmap) on top  — 32×32px, reliable fixed size
        - QLabel(text) on bottom      — 28px tall to comfortably hold 2 text lines
    The icon is now smaller (32 vs 44px) and the font larger (12 vs 10px) so
    labels are readable without being cramped.
    """

    def __init__(self, index, label, icon_pixmap=None, parent=None):
        super().__init__(parent)
        self.nav_index = index
        self._checked = False
        self.setMinimumSize(120, 86)  # taller to accommodate icon + two full text lines
        self.setObjectName("NavTopButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(label)              # full label on hover always visible

        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(3)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setFixedHeight(34)  # 34px for the icon
        if icon_pixmap is not None and not icon_pixmap.isNull():
            scaled = icon_pixmap.scaled(34, 34,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.icon_label.setPixmap(scaled)
        v.addWidget(self.icon_label)

        # Smart two-line label using the module-level format_nav_label()
        formatted = format_nav_label(label)
        self.text_label = QLabel(formatted)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.text_label.setWordWrap(True)   # allow wrapping as a safety net
        self.text_label.setMinimumHeight(38)  # enough for two lines at 13px + line spacing
        self.text_label.setStyleSheet(
            "font-size: 13px; font-weight: 500; background: transparent; "
            "line-height: 1.4; padding: 0px 2px;"
        )
        v.addWidget(self.text_label)

        self._update_style()

    def setChecked(self, checked):
        self._checked = checked
        self._update_style()

    def isChecked(self):
        return self._checked

    def _update_style(self):
        self.setProperty("checked", "true" if self._checked else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Emit via parent's handler
            if hasattr(self, '_click_callback'):
                self._click_callback(self.nav_index)
        super().mouseReleaseEvent(event)

# ==========================================
# JASP-STYLE DRAG & DROP VARIABLE WIDGETS
# ==========================================
VAR_MIME = "application/x-psystat-variable"

class VariableBank(QListWidget):
    """
    Universal 'Available Variables' source list, placed at the top of each tab's
    control column. Items can be dragged out onto any drop-enabled QListWidget
    (upgraded via enable_drag_drop_target) or QComboBox (via enable_drag_drop_combo)
    elsewhere in the same tab. The bank always lists every column in the dataset
    (or numeric-only columns, depending on the tab) -- it does not remove an item
    once it has been "used" downstream, since multiple analyses can reuse the same
    variable simultaneously (e.g. a variable can be both a covariate and a plotted
    axis at once).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setObjectName("VarBank")

    def startDrag(self, supportedActions):
        items = self.selectedItems()
        if not items:
            return
        names = [i.text() for i in items if i.flags() & Qt.ItemFlag.ItemIsEnabled]
        if not names:
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(VAR_MIME, ",".join(names).encode("utf-8"))
        mime.setText(",".join(names))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)



# ==========================================
# INTERACTIVE SEM GRAPHICS MODULE
# ==========================================
class SEMNode(QGraphicsItem):
    def __init__(self, name, is_latent=False):
        super().__init__()
        self.name = name
        self.is_latent = is_latent
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.edges = []
        self.width = 110
        self.height = 65 if not is_latent else 85

    def boundingRect(self):
        return QRectF(-self.width/2, -self.height/2, self.width, self.height)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        scene = self.scene()
        is_temp = False
        if scene and hasattr(scene, 'views') and scene.views():
            is_temp = getattr(scene.views()[0], 'temp_node', None) == self

        if is_temp: pen = QPen(QColor("#F59E0B"), 4)
        elif self.isSelected(): pen = QPen(QColor("#DC2626"), 3)
        else: pen = QPen(QColor("#2B6CB0") if self.is_latent else QColor("#4A5568"), 2)
            
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor("#EBF8FF") if self.is_latent else QColor("#FFFFFF")))

        if self.is_latent: painter.drawEllipse(self.boundingRect())
        else: painter.drawRoundedRect(self.boundingRect(), 5, 5)

        painter.setPen(QPen(Qt.GlobalColor.black))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, self.name)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)

class SEMEdge(QGraphicsItem):
    def __init__(self, source, dest):
        super().__init__()
        self.source = source
        self.dest = dest
        self.source.edges.append(self)
        self.dest.edges.append(self)
        self.setZValue(-1)
        self.arrowSize = 12
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    def update_position(self):
        self.prepareGeometryChange()

    def boundingRect(self):
        return QRectF(self.source.pos(), self.dest.pos()).normalized().adjusted(-15, -15, 15, 15)

    def paint(self, painter, option, widget):
        if not self.source or not self.dest: return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        line = QLineF(self.source.pos(), self.dest.pos())
        if line.length() == 0: return

        pen_color = QColor("#DC2626") if self.isSelected() else QColor("#4A5568")
        pen_width = 3 if self.isSelected() else 2
        painter.setPen(QPen(pen_color, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(line)

        angle = np.arctan2(-line.dy(), line.dx())
        dest_p = line.p2()
        
        offset = 45 if self.dest.is_latent else 35
        dest_p = QPointF(dest_p.x() - offset * np.cos(angle), dest_p.y() + offset * np.sin(angle))

        arrowP1 = dest_p + QPointF(np.sin(angle - np.pi / 3) * self.arrowSize, np.cos(angle - np.pi / 3) * self.arrowSize)
        arrowP2 = dest_p + QPointF(np.sin(angle - np.pi + np.pi / 3) * self.arrowSize, np.cos(angle - np.pi + np.pi / 3) * self.arrowSize)
        
        arrowHead = QPolygonF([dest_p, arrowP1, arrowP2])
        painter.setBrush(pen_color)
        painter.drawPolygon(arrowHead)

class SEMCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 1000, 800)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.mode = "Select"
        self.temp_node = None
        self.nodes = []
        self.edges = []

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())
        item = self.scene.itemAt(pos, self.transform())
        
        if self.mode == "Path":
            node = None
            if isinstance(item, SEMNode):
                node = item
            elif item is not None:
                parent = item.parentItem()
                while parent:
                    if isinstance(parent, SEMNode):
                        node = parent
                        break
                    parent = parent.parentItem()

            if node:
                if not self.temp_node:
                    self.temp_node = node
                    self.scene.update() 
                else:
                    if self.temp_node != node:
                        edge = SEMEdge(self.temp_node, node)
                        self.scene.addItem(edge)
                        self.edges.append(edge)
                        edge.update_position()
                    
                    self.temp_node = None
                    self.scene.update() 
        else:
            super().mousePressEvent(event)

    def add_node(self, name, is_latent, pos=None):
        node = SEMNode(name, is_latent)
        if pos:
            node.setPos(pos)
        else:
            offset = (len(self.nodes) % 10) * 30
            node.setPos(150 + offset, 150 + offset)
        self.scene.addItem(node)
        self.nodes.append(node)
        return node

    def delete_selected(self):
        for item in self.scene.selectedItems():
            if isinstance(item, SEMNode):
                for edge in list(item.edges):
                    if edge in self.scene.items():
                        self.scene.removeItem(edge)
                        if edge in self.edges: self.edges.remove(edge)
                        if edge.source == item and edge in edge.dest.edges: edge.dest.edges.remove(edge)
                        elif edge.dest == item and edge in edge.source.edges: edge.source.edges.remove(edge)
                self.scene.removeItem(item)
                if item in self.nodes: self.nodes.remove(item)
            elif isinstance(item, SEMEdge):
                if item in item.source.edges: item.source.edges.remove(item)
                if item in item.dest.edges: item.dest.edges.remove(item)
                self.scene.removeItem(item)
                if item in self.edges: self.edges.remove(item)

    def generate_syntax(self):
        measurement = {}
        structural = {}
        for edge in self.edges:
            src = edge.source
            dst = edge.dest
            if src.is_latent and not dst.is_latent:
                if src.name not in measurement: measurement[src.name] = []
                measurement[src.name].append(dst.name)
            else:
                if dst.name not in structural: structural[dst.name] = []
                structural[dst.name].append(src.name)
                
        res = []
        if measurement:
            res.append("# Measurement Model")
            for k, v in measurement.items():
                res.append(f"{k} =~ " + " + ".join(v))
        if structural:
            if res: res.append("")
            res.append("# Structural Model")
            for k, v in structural.items():
                res.append(f"{k} ~ " + " + ".join(v))
                
        return "\n".join(res)

    def load_example(self):
        self.scene.clear()
        self.nodes.clear()
        self.edges.clear()
        self.temp_node = None
        
        d = self.add_node("Depression", True, QPointF(250, 150))
        a = self.add_node("Anxiety", True, QPointF(550, 150))
        
        i1 = self.add_node("item1", False, QPointF(150, 350))
        i2 = self.add_node("item2", False, QPointF(350, 350))
        
        e1 = SEMEdge(d, i1); self.scene.addItem(e1); self.edges.append(e1)
        e2 = SEMEdge(d, i2); self.scene.addItem(e2); self.edges.append(e2)
        e3 = SEMEdge(d, a); self.scene.addItem(e3); self.edges.append(e3)
        
        for edge in [e1, e2, e3]: edge.update_position()

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==========================================
# MAIN APPLICATION
# ==========================================
class PsyStat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PsyStat - Statistical Environment")
        self.setWindowIcon(_make_psystat_qicon())
        # Screen-aware sizing — avoids "Unable to set geometry" on high-DPI displays
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1400, int(avail.width() * 0.92))
            h = min(900, int(avail.height() * 0.92))
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2,
                      avail.y() + (avail.height() - h) // 2)
        else:
            self.resize(1400, 900)
        self.setWindowIcon(_make_psystat_qicon())
        
        self.df = None
        self.history = []
        self.redo_stack = []
        self.MAX_UNDO_HISTORY = 15  # cap full-DataFrame snapshots to bound memory use
        self.decimals = 3
        self.is_dark_mode = False
        
        self.var_labels = {}
        self.var_value_labels = {}
        self.var_scales = {}

        self.setup_menu()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.nav_items = [
            "Data Management", 
            "Data Visualization",
            "Descriptives & Crosstabs", 
            "Item Analysis & CVI",
            "Correlation",
            "Compare Means", 
            "Analysis of Variance", 
            "Regression", 
            "Mediation Analysis",
            "Categorical PCA (CATPCA)",
            "Cluster & Profile Analysis (LCA)",
            "Forecasting (Time Series)",
            "Exploratory (EFA)", 
            "Confirmatory (CFA)",
            "SEM (Graph & Syntax)",
            "Network Analysis",
            "Power Analysis"
        ]

        # --- Two-row top navigation bar. Each of the 17 module buttons is a
        # NavTopButton: a custom QWidget with a 34px matplotlib icon above a
        # two-line text label. Arranged in two rows of 9 columns (9 + 8 = 17),
        # so the full menu is visible at once on a 1366px+ wide screen without
        # any scrolling. The bar height (~190px) is proportional to the window
        # below -- it takes ~21% of a 900px-tall window, leaving the analysis
        # panels the other 79%. ---
        nav_bar_container = QWidget()
        nav_bar_container.setObjectName("NavBarContainer")
        nav_bar_outer = QVBoxLayout(nav_bar_container)
        nav_bar_outer.setContentsMargins(4, 4, 4, 0)
        nav_bar_outer.setSpacing(0)

        nav_grid_widget = QWidget()
        nav_grid = QGridLayout(nav_grid_widget)
        nav_grid.setContentsMargins(4, 4, 4, 4)
        nav_grid.setHorizontalSpacing(3)
        nav_grid.setVerticalSpacing(3)
        COLS = 9  # 9 + 8 = 17 items across two rows

        self.nav_buttons = []
        for i, item in enumerate(self.nav_items):
            icon_pixmap = generate_nav_icon(item, dark=self.is_dark_mode, size_px=48, dpi=100)
            btn = NavTopButton(i, item, icon_pixmap)
            btn._click_callback = self.display_tab
            self.nav_buttons.append(btn)
            row, col = divmod(i, COLS)
            nav_grid.addWidget(btn, row, col)

        # Fill remaining cells in the last row so buttons stretch proportionally
        for fill_col in range(len(self.nav_items) % COLS, COLS):
            spacer = QWidget()
            spacer.setMinimumHeight(86)
            nav_grid.addWidget(spacer, (len(self.nav_items) - 1) // COLS, fill_col)

        # All columns share width equally so buttons stretch with the window
        for c in range(COLS):
            nav_grid.setColumnStretch(c, 1)

        nav_bar_outer.addWidget(nav_grid_widget)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("NavBarSeparator")
        sep.setFixedHeight(2)
        nav_bar_outer.addWidget(sep)

        self.main_layout.addWidget(nav_bar_container)

        self.tabs = QStackedWidget()
        self.main_layout.addWidget(self.tabs, stretch=1)

        self.init_data_tab()
        self.init_viz_tab()
        self.init_descriptives_tab()
        self.init_item_analysis_tab()
        self.init_correlation_tab()
        self.init_compare_means_tab()
        self.init_anova_tab()
        self.init_regression_tab()
        self.init_mediation_tab()
        self.init_catpca_tab()
        self.init_lca_tab()
        self.init_forecast_tab()
        self.init_efa_tab()
        self.init_cfa_tab()
        self.init_sem_tab()
        self.init_sna_tab()
        self.init_power_tab()

        self.apply_modern_theme()
        self.display_tab(0)

        # --- Crash recovery / autosave ---
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(120_000)  # 2 minutes
        self.autosave_timer.timeout.connect(self.autosave_session)
        self.autosave_timer.start()
        # Defer the recovery check until after the window is shown, so the prompt
        # doesn't appear before the user can see the app exists.
        QTimer.singleShot(300, self.check_for_crash_recovery)

    def get_autosave_path(self):
        import tempfile
        base = os.path.join(tempfile.gettempdir(), "psystat_autosave")
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            pass
        return os.path.join(base, "session_recovery.pkl")

    def autosave_session(self):
        """
        Periodically snapshots the current dataset + variable metadata to a temp
        file so an unexpected crash doesn't lose an entire working session. This is
        a lightweight safety net, not a substitute for the user explicitly saving
        their dataset/results -- it only stores the data and labels, not analysis
        results or undo history, to keep autosave fast and unobtrusive.
        """
        if self.df is None:
            return
        try:
            import pickle, time
            payload = {
                'df': self.df,
                'var_labels': getattr(self, 'var_labels', {}),
                'var_scales': getattr(self, 'var_scales', {}),
                'var_value_labels': getattr(self, 'var_value_labels', {}),
                'timestamp': time.time(),
            }
            with open(self.get_autosave_path(), 'wb') as f:
                pickle.dump(payload, f)
        except Exception:
            pass  # Autosave must never interrupt the user's work with an error dialog.

    def check_for_crash_recovery(self):
        """
        On startup, checks whether a recent autosave snapshot exists (left behind by
        an unclean exit -- a clean exit deletes it in closeEvent) and offers to
        restore it. Snapshots older than 24 hours are treated as stale and ignored
        rather than offered, since the data was very likely already handled.
        """
        try:
            import pickle, time
            path = self.get_autosave_path()
            if not os.path.exists(path):
                return
            with open(path, 'rb') as f:
                payload = pickle.load(f)
            age_hours = (time.time() - payload.get('timestamp', 0)) / 3600
            if age_hours > 24:
                os.remove(path)
                return
            reply = QMessageBox.question(
                self, "Recover Previous Session?",
                f"It looks like PsyStat closed unexpectedly. A saved session from "
                f"{age_hours*60:.0f} minutes ago is available "
                f"({len(payload['df'])} rows, {len(payload['df'].columns)} columns).\n\n"
                f"Would you like to recover it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.df = payload['df']
                self.var_labels = payload.get('var_labels', {})
                self.var_scales = payload.get('var_scales', {})
                self.var_value_labels = payload.get('var_value_labels', {})
                self.update_global_dropdowns()
                self.populate_data_tables()
            os.remove(path)
        except Exception:
            pass  # A corrupted/unreadable recovery file should never block startup.

    def closeEvent(self, event):
        """On a clean exit, remove the autosave file so the next launch doesn't
        offer to 'recover' a session that was already closed intentionally."""
        try:
            path = self.get_autosave_path()
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        event.accept()

    # ==========================================
    # LAVAAN FORMATTER & HELPERS
    # ==========================================
    def calculate_srmr(self, model, num_df):
        try:
            sigma_res = model.calc_sigma()
            Sigma = np.array(sigma_res[0]) if isinstance(sigma_res, tuple) else np.array(sigma_res)
            
            if hasattr(model, 'mx_cov') and model.mx_cov is not None:
                S = np.array(model.mx_cov)
            else:
                return np.nan, "Sample covariance matrix 'mx_cov' not found in model."
            
            if S.shape != Sigma.shape:
                return np.nan, f"Shape mismatch: Empirical S {S.shape} vs Implied Sigma {Sigma.shape}"
            
            diff = S - Sigma
            std_devs = np.sqrt(np.diag(S))
            std_devs[std_devs == 0] = 1e-9 
            std_matrix = np.outer(std_devs, std_devs)
            std_res = diff / std_matrix
            
            tril_idx = np.tril_indices_from(std_res)
            tril_res = std_res[tril_idx]
            srmr = np.sqrt(np.mean(tril_res ** 2))
            
            return srmr, ""
        except Exception as e:
            return np.nan, f"SRMR Engine Error: {str(e)}"

    def generate_sem_html_report(self, model, stats, ins, std_ins, num_obs):
        def get_val(col_list, default=np.nan):
            for c in col_list:
                if c in stats.columns: return stats[c].values[0]
            return default
            
        chi2 = get_val(['chi2', 'Chi2'])
        dof = get_val(['DoF', 'df'])
        pval = get_val(['chi2 p-value', 'p-value', 'pvalue'])
        chi2_b = get_val(['chi2 Baseline', 'chi2 baseline'])
        dof_b = get_val(['DoF Baseline', 'df baseline'])
        pval_b = get_val(['Baseline p-value', 'chi2 Baseline p-value'])
        # semopy's calc_stats() does not actually output a baseline p-value column
        # under any of these names in any released version (verified against its
        # source: semopy.com/docs/stats.html) -- it only returns 'chi2 Baseline' and
        # 'DoF Baseline'. The lookup above will therefore always return NaN. Since
        # the baseline chi-square statistic follows a chi-square distribution with
        # dof_b degrees of freedom under the null (same logic semopy itself uses for
        # the user model's own chi2 p-value), we can compute it directly instead of
        # leaving it blank.
        if pd.isna(pval_b) and not pd.isna(chi2_b) and not pd.isna(dof_b) and dof_b > 0:
            try:
                pval_b = stats.chi2.sf(chi2_b, dof_b)
            except Exception:
                pval_b = np.nan
        cfi = get_val(['CFI'])
        tli = get_val(['TLI'])
        loglik = get_val(['LogLik', 'loglik'])
        aic = get_val(['AIC'])
        bic = get_val(['BIC'])
        rmsea = get_val(['RMSEA'])
        
        opt_method = "NLMINB"
        if hasattr(model, 'opt_method'): opt_method = model.opt_method.upper()

        srmr = get_val(['SRMR', 'srmr'])
        if pd.isna(srmr): srmr, _ = self.calculate_srmr(model, self.df.select_dtypes(include=[np.number]).dropna())

        out = self.get_apa_css()
        
        out += "<h2>Model Information</h2><table class='apa'><tr><th>Property</th><th>Value</th></tr>"
        out += f"<tr><td style='text-align:left;'>Estimator</td><td>Maximum Likelihood (ML)</td></tr>"
        out += f"<tr><td style='text-align:left;'>Optimization Method</td><td>{opt_method}</td></tr>"
        out += f"<tr><td style='text-align:left;'>Number of Model Parameters</td><td>{len(ins)}</td></tr>"
        out += f"<tr><td style='text-align:left;'>Number of Observations</td><td>{num_obs}</td></tr>"
        out += "</table>"
        
        out += "<h2>Model Fit Indices</h2><table class='apa'><tr><th>Metric</th><th>Value</th></tr>"
        
        p_str = f"<span class='sig'>{self.fmt(pval, True)}</span>" if not pd.isna(pval) and pval < 0.05 else self.fmt(pval, True)
        out += f"<tr><td style='text-align:left;'><b>Test Statistic (User Model χ²)</b></td><td>{self.fmt(chi2)}</td></tr>"
        out += f"<tr><td style='text-align:left;'>Degrees of Freedom (df)</td><td>{self.fmt(dof)}</td></tr>"
        out += f"<tr><td style='text-align:left;'>P-value (Chi-square)</td><td>{p_str}</td></tr>"
        
        if not pd.isna(chi2_b):
            out += f"<tr><td style='text-align:left;'><b>Test Statistic (Baseline Model χ²)</b></td><td>{self.fmt(chi2_b)}</td></tr>"
            out += f"<tr><td style='text-align:left;'>Baseline Degrees of Freedom</td><td>{self.fmt(dof_b)}</td></tr>"
            out += f"<tr><td style='text-align:left;'>Baseline P-value</td><td>{self.fmt(pval_b, True)}</td></tr>"
            
        out += f"<tr><td style='text-align:left;'><b>Comparative Fit Index (CFI)</b></td><td>{self.fmt(cfi)}</td></tr>"
        out += f"<tr><td style='text-align:left;'><b>Tucker-Lewis Index (TLI)</b></td><td>{self.fmt(tli)}</td></tr>"
        out += f"<tr><td style='text-align:left;'>Loglikelihood user model (H0)</td><td>{self.fmt(loglik)}</td></tr>"
        out += f"<tr><td style='text-align:left;'>Akaike (AIC)</td><td>{self.fmt(aic)}</td></tr>"
        out += f"<tr><td style='text-align:left;'>Bayesian (BIC)</td><td>{self.fmt(bic)}</td></tr>"
        out += f"<tr><td style='text-align:left;'><b>RMSEA</b></td><td>{self.fmt(rmsea)}</td></tr>"
        out += f"<tr><td style='text-align:left;'><b>SRMR</b></td><td>{self.fmt(srmr)}</td></tr>"
        out += "</table>"
        
        # ── Build standardized estimate lookup ────────────────────────────────
        # Three possible sources, tried in order of accuracy:
        #   1. semopy inspect(std_est=True) → 'Est. Std' column (Std.all)
        #      This is the fully-standardized solution (both LV and observed
        #      variances set to 1), equivalent to lavaan's std.all.
        #   2. semopy inspect(std_est='lv') if available → Std.lv
        #      Standardizes only latent variances; observed metric preserved.
        #   3. Fallback: manually compute both from unstandardized estimates
        #      using model parameter matrices (Sigma, implied covariances).

        # Attempt native std_est='lv' for Std.lv
        try:
            ins_lv = model.inspect(std_est='lv')
            has_stdlv = 'Est. Std' in ins_lv.columns
        except Exception:
            ins_lv = pd.DataFrame()
            has_stdlv = False

        # Attempt native std_est=True for Std.all
        try:
            ins_all = model.inspect(std_est=True)
            has_stdall = 'Est. Std' in ins_all.columns
        except Exception:
            ins_all = pd.DataFrame()
            has_stdall = False

        # If native std_est not available, compute manually from model matrices
        if not has_stdall:
            try:
                # Get implied covariance matrix (Sigma) from the fitted model
                obs_vars = [c for c in num_df.columns if c in ins['rval'].values or c in ins['lval'].values]
                sigma = pd.DataFrame(model.calculate_sigma()[0],
                                     index=obs_vars[:model.calculate_sigma()[0].shape[0]],
                                     columns=obs_vars[:model.calculate_sigma()[0].shape[0]])
                # Build per-row Std.all: β_std = β_unst * (SD_x / SD_y)
                # where SD is taken from the diagonal of Sigma for observed,
                # or from the latent variance estimates for latent variables.
                lv_var = {}
                for _, r in ins[ins['op'] == '~~'].iterrows():
                    if r['lval'] == r['rval']:
                        lv_var[r['lval']] = float(r['Estimate'])

                def _sd(name):
                    if name in sigma.columns:
                        v = sigma.loc[name, name]
                        return float(np.sqrt(v)) if v > 0 else 1.0
                    return float(np.sqrt(lv_var.get(name, 1.0))) if lv_var.get(name, 1.0) > 0 else 1.0

                std_all_map = {}
                std_lv_map  = {}
                for _, r in ins.iterrows():
                    key = (r['lval'], r['op'], r['rval'])
                    b   = float(r['Estimate'])
                    sd_x = _sd(r['rval'])
                    sd_y = _sd(r['lval'])
                    sd_lv_y = float(np.sqrt(lv_var.get(r['lval'], sd_y**2)))
                    std_all_map[key] = b * sd_x / sd_y  if sd_y != 0 else np.nan
                    std_lv_map[key]  = b * sd_x / sd_lv_y if sd_lv_y != 0 else np.nan
                has_stdall = True
                has_stdlv  = True
            except Exception:
                std_all_map = {}
                std_lv_map  = {}

        def _get_std(row, which='all'):
            lval, op, rval = row['lval'], row['op'], row['rval']
            # Native path
            if which == 'all' and has_stdall and not ins_all.empty:
                m = ins_all[(ins_all['lval'] == lval) & (ins_all['op'] == op) & (ins_all['rval'] == rval)]
                if not m.empty: return self.fmt(m['Est. Std'].values[0])
            if which == 'lv' and has_stdlv and not ins_lv.empty:
                m = ins_lv[(ins_lv['lval'] == lval) & (ins_lv['op'] == op) & (ins_lv['rval'] == rval)]
                if not m.empty: return self.fmt(m['Est. Std'].values[0])
            # Fallback computed maps
            key = (lval, op, rval)
            if which == 'all' and key in std_all_map: return self.fmt(std_all_map[key])
            if which == 'lv'  and key in std_lv_map:  return self.fmt(std_lv_map[key])
            # std_ins from z-score fallback (legacy)
            if not std_ins.empty:
                m = std_ins[(std_ins['lval'] == lval) & (std_ins['op'] == op) & (std_ins['rval'] == rval)]
                if not m.empty: return self.fmt(m['Estimate'].values[0])
            return "-"

        show_std = has_stdall or has_stdlv or not std_ins.empty

        out += "<h2>Parameter Estimates</h2>"
        out += ("<div style='background:#EFF6FF; border-left:4px solid #3B82F6; padding:8px 12px; "
                "margin:0 0 10px 0; font-size:12px; color:#1E3A8A;'>"
                "<b>Estimate</b> = unstandardized (raw metric). &nbsp;"
                "<b>Std.lv</b> = standardized for latent variables only (latent SD = 1; observed metric preserved). &nbsp;"
                "<b>Std.all</b> = fully standardized (all SD = 1; comparable to a correlation). &nbsp;"
                "SE, z, and p apply to the unstandardized estimate."
                "</div>")

        def render_param_table(title, op, condition=None):
            subset = ins[ins['op'] == op].copy()
            if condition: subset = subset[condition(subset)]
            if subset.empty: return ""

            html = f"<h3>{title}</h3><table class='apa'>"
            if show_std:
                html += ("<tr>"
                         "<th>LHS</th><th>op</th><th>RHS</th>"
                         "<th>Estimate</th><th>Std.Err</th><th>z-value</th><th>P(&gt;|z|)</th>"
                         "<th style='border-left:2px solid #CBD5E1;'>Std.lv</th>"
                         "<th>Std.all</th>"
                         "</tr>")
            else:
                html += ("<tr><th>LHS</th><th>op</th><th>RHS</th>"
                         "<th>Estimate</th><th>Std.Err</th><th>z-value</th><th>P(&gt;|z|)</th></tr>")

            for _, row in subset.iterrows():
                lval = row['lval']
                rval = row['rval']
                est  = self.fmt(row['Estimate'])
                se   = self.fmt(row.get('Std. Err', np.nan))
                z    = self.fmt(row.get('z-value', np.nan))
                p    = row.get('p-value', np.nan)
                p_str = (f"<span class='sig'>{self.fmt(p, True)}</span>"
                         if not pd.isna(p) and isinstance(p, (int, float)) and p < 0.05
                         else self.fmt(p, True))

                # Highlight large standardized loadings (≥ .50) in Latent Variables table
                is_loading = (op == '=~')

                if show_std:
                    stdlv_raw  = _get_std(row, 'lv')
                    stdall_raw = _get_std(row, 'all')

                    def _hl(val_str):
                        """Bold if numeric and ≥ .50 (conventional salient loading threshold)."""
                        try:
                            if is_loading and abs(float(val_str)) >= 0.50:
                                return f"<b>{val_str}</b>"
                        except (ValueError, TypeError):
                            pass
                        return val_str

                    html += (f"<tr>"
                             f"<td style='text-align:left;'><b>{lval}</b></td>"
                             f"<td>{op}</td>"
                             f"<td style='text-align:left;'>{rval}</td>"
                             f"<td>{est}</td><td>{se}</td><td>{z}</td><td>{p_str}</td>"
                             f"<td style='border-left:2px solid #CBD5E1;'>{_hl(stdlv_raw)}</td>"
                             f"<td>{_hl(stdall_raw)}</td>"
                             f"</tr>")
                else:
                    html += (f"<tr>"
                             f"<td style='text-align:left;'><b>{lval}</b></td>"
                             f"<td>{op}</td>"
                             f"<td style='text-align:left;'>{rval}</td>"
                             f"<td>{est}</td><td>{se}</td><td>{z}</td><td>{p_str}</td>"
                             f"</tr>")

            html += "</table>"
            return html

        out += render_param_table("Latent Variables (Factor Loadings)", "=~")
        out += render_param_table("Regressions", "~")
        out += render_param_table("Covariances", "~~", lambda df: df['lval'] != df['rval'])
        out += render_param_table("Intercepts", "~1")
        out += render_param_table("Variances", "~~", lambda df: df['lval'] == df['rval'])

        return out

    def generate_standardized_residuals_html(self, model):
        out = "<h2>📐 Modification Indices (Standardized Residual Proxy)</h2>"
        out += ("<div style='background:#EFF6FF; border-left:4px solid #3B82F6; padding:10px 14px; "
                "margin:8px 0; font-size:13px; color:#1E3A8A; line-height:1.5;'>"
                "This is the largest-standardized-residual approach to modification indices: pairs of "
                "variables whose observed covariance the model badly under- or over-predicts are flagged as "
                "candidates for adding a path or covariance between them. <code>semopy</code> does not provide "
                "true Lagrange-multiplier (score test) modification indices the way lavaan or Mplus do, so this "
                "is reported as a residual-based proxy rather than a formal χ²(1) score statistic — it identifies "
                "the <i>same</i> problem areas in practice, but the numbers are standardized residuals, not "
                "score-test statistics, so do not directly compare them to lavaan/Mplus MI values. "
                "The table below always lists the largest residuals regardless of significance, since even a "
                "non-significant residual can flag the most promising place to improve the model.</div>")

        try:
            sigma_res = model.calc_sigma()
            Sigma = np.array(sigma_res[0]) if isinstance(sigma_res, tuple) else np.array(sigma_res)
            S = np.array(model.mx_cov)
            diff = S - Sigma

            std_devs = np.sqrt(np.diag(S))
            std_devs[std_devs == 0] = 1e-9
            std_matrix = np.outer(std_devs, std_devs)
            std_res = diff / std_matrix

            var_names = []

            # NOTE: calc_sigma() returns (Sigma, (m, c)) where (m, c) are AUXILIARY
            # MATRICES used internally for gradient computations -- NOT variable names
            # (confirmed against semopy's own source: semopy.com/docs/model.html). The
            # second element of that tuple can never be used as a name list, so that
            # possibility is intentionally not attempted here (a previous version of
            # this code incorrectly tried sigma_res[1], which silently always failed
            # and fell through to the methods below -- harmless, but worth removing).

            # Method 1: Extract from model.vars dict
            if hasattr(model, 'vars') and isinstance(model.vars, dict) and 'observed' in model.vars:
                var_names = list(model.vars['observed'])
                
            # Method 2: Check mx_cov columns
            if (not var_names or len(var_names) != len(std_res)) and hasattr(model, 'mx_cov') and hasattr(model.mx_cov, 'columns'):
                var_names = model.mx_cov.columns.tolist()
                
            # Method 3: Check model.observed directly
            if (not var_names or len(var_names) != len(std_res)) and hasattr(model, 'observed'):
                var_names = list(model.observed)
            
            # Method 4: Fallback if all mapping fails
            is_generic = False
            if not var_names or len(var_names) != len(std_res):
                var_names = [f"Item_{i+1}" for i in range(len(std_res))]
                is_generic = True

            if len(var_names) == len(std_res):
                res_records = []
                for i in range(len(var_names)):
                    for j in range(i + 1, len(var_names)):
                        val = std_res[i, j]
                        if not np.isnan(val):
                            res_records.append({
                                'Var 1': var_names[i],
                                'Var 2': var_names[j],
                                'Residual': val,
                                'Abs_Residual': abs(val)
                            })

                if res_records:
                    if is_generic:
                        out += f"<p class='warn' style='font-size:12px; margin-bottom:10px;'>Note: Could not perfectly map variable names from the model environment. Displaying generic structural labels.</p>"

                    res_df = pd.DataFrame(res_records).sort_values(by='Abs_Residual', ascending=False).head(15)
                    n_significant = int((res_df['Abs_Residual'] > 1.96).sum())

                    if n_significant == 0:
                        out += (
                            "<p style='color:#10B981; font-weight:bold;'><i>No standardized residual exceeds the "
                            "conventional |1.96| significance threshold — by that formal criterion, no single pair "
                            "stands out as a statistically clear modification target. The table below still shows "
                            "the largest residuals in the model so you can judge for yourself whether any are "
                            "practically worth investigating, even below that threshold.</i></p>"
                        )
                    out += "<table class='apa'><tr><th>Primary Variable</th><th>Missing Connection</th><th>Standardized Residual</th><th>Recommendation</th></tr>"
                    for _, row in res_df.iterrows():
                        val = row['Residual']
                        is_sig = abs(val) > 1.96
                        color = "#EF4444" if is_sig else "#D97706"
                        rec = "Consider adding a path or covariance" if is_sig else "Below significance threshold; minor candidate"
                        out += f"<tr><td style='text-align:left;'><b>{row['Var 1']}</b></td><td style='text-align:left;'><b>{row['Var 2']}</b></td><td style='color:{color}; font-weight:bold;'>{self.fmt(val)}</td><td>{rec}</td></tr>"
                    out += "</table>"
                else:
                    out += "<p style='color:#6B7280;'><i>No off-diagonal residuals could be computed (the model may have only one observed variable per factor, leaving no covariance pairs to evaluate).</i></p>"
            else:
                out += "<p class='warn'>Residuals computed, but could not map variable names to the matrix.</p>"
        except Exception as e:
            out += f"<p class='warn'>Could not calculate residuals: {str(e)}</p>"
            
        return out

    def plot_smooth_fit(self, ax, x, y, color):
        """Helper to plot smoothed lines with 95% Confidence Intervals."""
        if len(x) < 3 or x.std() == 0: return
        
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_cl = x[mask].values if isinstance(x, pd.Series) else x[mask]
        y_cl = y[mask].values if isinstance(y, pd.Series) else y[mask]
        
        m, b = np.polyfit(x_cl, y_cl, 1)
        x_line = np.linspace(x_cl.min(), x_cl.max(), 100)
        y_line = m * x_line + b
        
        n = len(x_cl)
        if n > 2:
            t_val = stats.t.ppf(1 - 0.05/2, n - 2)
            resid = y_cl - (m * x_cl + b)
            s_err = np.sqrt(np.sum(resid**2) / (n - 2))
            mean_x = np.mean(x_cl)
            sum_sq_x = np.sum((x_cl - mean_x)**2)
            
            ci = t_val * s_err * np.sqrt(1/n + (x_line - mean_x)**2 / sum_sq_x)
            
            ax.plot(x_line, y_line, color=color, linewidth=2.5)
            ax.fill_between(x_line, y_line - ci, y_line + ci, color=color, alpha=0.2)

    def build_residual_normality_plots(self, residuals):
        """
        SPSS-style residual normality diagnostics: a Normal P-P Plot (observed vs.
        expected cumulative probability of the standardized residuals) side-by-side
        with a Normal Q-Q Plot (theoretical vs. sample quantiles). Both carry a 45°
        reference line so departure from normality can be eyeballed directly, exactly
        like SPSS's "Normal P-P Plot of Regression Standardized Residual". The
        Shapiro-Wilk result is annotated on the figure so the numeric test and the
        visual check are read together rather than in isolation.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        res = np.asarray(residuals, dtype=float)
        res = res[~np.isnan(res)]
        if len(res) < 3:
            return None
        std_res = (res - res.mean()) / res.std(ddof=1)

        is_dark = self.is_dark_mode
        text_c = 'white' if is_dark else 'black'
        point_c = '#818CF8' if is_dark else '#4F46E5'
        line_c = '#EF4444'

        fig = Figure(figsize=(10, 4.6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        if is_dark:
            fig.patch.set_facecolor('#1F2937')
            ax1.set_facecolor('#1F2937'); ax2.set_facecolor('#1F2937')

        # --- Normal P-P Plot ---
        sorted_res = np.sort(std_res)
        n_obs = len(sorted_res)
        observed_cp = (np.arange(1, n_obs + 1) - 0.5) / n_obs
        expected_cp = stats.norm.cdf(sorted_res)
        ax1.scatter(expected_cp, observed_cp, s=14, color=point_c, alpha=0.7, edgecolors='none')
        ax1.plot([0, 1], [0, 1], color=line_c, linewidth=1.6)
        ax1.set_xlabel("Expected Cumulative Probability", color=text_c)
        ax1.set_ylabel("Observed Cumulative Probability", color=text_c)
        ax1.set_title("Normal P-P Plot of\nRegression Standardized Residual", fontsize=10.5, fontweight='bold', color=text_c)
        ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
        ax1.tick_params(colors=text_c)

        # --- Normal Q-Q Plot ---
        (osm, osr), (slope, intercept, r) = stats.probplot(std_res, dist="norm")
        ax2.scatter(osm, osr, s=14, color=point_c, alpha=0.7, edgecolors='none')
        ax2.plot(osm, slope * osm + intercept, color=line_c, linewidth=1.6)
        ax2.set_xlabel("Theoretical Quantiles", color=text_c)
        ax2.set_ylabel("Sample Quantiles (Std. Residuals)", color=text_c)
        ax2.set_title(f"Normal Q-Q Plot of Residuals\nR² = {r**2:.3f}", fontsize=10.5, fontweight='bold', color=text_c)
        ax2.tick_params(colors=text_c)

        for ax in (ax1, ax2):
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        fig.tight_layout()
        chart = self.make_zoomable_chart(fig)

        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)

        shapiro_w, shapiro_p = (np.nan, np.nan)
        if 3 <= len(res) <= 5000:
            try:
                shapiro_w, shapiro_p = stats.shapiro(res)
            except Exception:
                pass

        if not np.isnan(shapiro_p):
            verdict = "does not significantly deviate from normality" if shapiro_p > 0.05 else "significantly deviates from normality"
            badge_color = "#10B981" if shapiro_p > 0.05 else "#EF4444"
            note_html = (f"<b>Shapiro-Wilk:</b> W = {self.fmt(shapiro_w)}, p = {self.fmt(shapiro_p, True)} "
                         f"— residual distribution <span style='color:{badge_color}; font-weight:bold;'>{verdict}</span>. "
                         f"Points hugging the red reference line in both plots above confirm this visually; "
                         f"systematic curvature or S-shapes indicate skew or heavy/light tails.")
        else:
            note_html = "<i>Shapiro-Wilk requires between 3 and 5000 residuals; use the plots above to assess normality visually for this sample size.</i>"

        note = QLabel(note_html)
        note.setObjectName("InfoLabel")
        note.setWordWrap(True)
        wl.addWidget(note)
        wl.addWidget(chart)
        return wrapper

    def build_group_descriptives_table(self, data, dvs, group_col):
        """
        Builds a Mean/SD/N descriptive statistics table broken down by group, for
        one or more dependent variables. Used by ANOVA, MANOVA, and Repeated
        Measures results so the reader can see the actual group differences behind
        the F-test, not just the test statistic -- the standard "Descriptive
        Statistics" table every published ANOVA results section includes.
        """
        try:
            groups = sorted(data[group_col].dropna().unique(), key=lambda x: str(x))
        except Exception:
            groups = list(data[group_col].dropna().unique())

        out = "<h2>Descriptive Statistics</h2><table class='apa'><tr><th>Dependent Variable</th><th>" + str(group_col) + "</th><th>N</th><th>Mean</th><th>SD</th></tr>"
        for dv in dvs:
            for i, g in enumerate(groups):
                sub = data[data[group_col] == g][dv].dropna()
                label = f"<b>{dv}</b>" if i == 0 else ""
                out += f"<tr><td style='text-align:left;'>{label}</td><td style='text-align:left;'>{g}</td><td>{len(sub)}</td><td>{self.fmt(sub.mean())}</td><td>{self.fmt(sub.std())}</td></tr>"
            total = data[dv].dropna()
            out += f"<tr><td style='text-align:left;'></td><td style='text-align:left;'><i>Total</i></td><td>{len(total)}</td><td>{self.fmt(total.mean())}</td><td>{self.fmt(total.std())}</td></tr>"
        out += "</table>"
        return out

    def build_repeated_measures_descriptives_table(self, data, measures):
        """
        Builds a Mean/SD/N descriptive statistics table for repeated-measures data,
        one row per timepoint/condition column (rather than per group, since
        repeated measures has no grouping variable -- every subject contributes to
        every row).
        """
        out = "<h2>Descriptive Statistics</h2><table class='apa'><tr><th>Timepoint / Condition</th><th>N</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th></tr>"
        for m in measures:
            d = data[m].dropna()
            out += f"<tr><td style='text-align:left;'><b>{m}</b></td><td>{len(d)}</td><td>{self.fmt(d.mean())}</td><td>{self.fmt(d.std())}</td><td>{self.fmt(d.min())}</td><td>{self.fmt(d.max())}</td></tr>"
        out += "</table>"
        return out

    def build_pairwise_corrected_table(self, data, measures, paired=True, nonparametric=False, correction='bonferroni'):
        """
        Runs every pairwise comparison among 3+ `measures` columns in `data` and
        applies a multiple-comparisons correction (Bonferroni or Benjamini-Hochberg
        FDR) to the resulting p-values, returning an HTML table. Used as the
        standard post-hoc step after a significant omnibus RM-ANOVA/Friedman test,
        since running k(k-1)/2 uncorrected tests inflates the family-wise Type I
        error rate well past the nominal alpha.
        `paired=True` assumes the same subjects across columns (t-test/Wilcoxon on
        differences); `nonparametric=True` uses Wilcoxon signed-rank instead of a
        paired t-test.
        """
        from itertools import combinations
        pairs = list(combinations(measures, 2))
        if not pairs:
            return "<p style='color:#6B7280;'><i>Not enough conditions for pairwise comparisons.</i></p>"

        raw_pvals = []
        rows_data = []
        for m1, m2 in pairs:
            sub = data[[m1, m2]].dropna()
            if len(sub) < 3:
                raw_pvals.append(np.nan)
                rows_data.append((m1, m2, np.nan, np.nan, len(sub)))
                continue
            try:
                if nonparametric:
                    stat, p = stats.wilcoxon(sub[m1], sub[m2])
                    stat_name = "W"
                elif paired:
                    stat, p = stats.ttest_rel(sub[m1], sub[m2])
                    stat_name = "t"
                else:
                    stat, p = stats.ttest_ind(sub[m1], sub[m2])
                    stat_name = "t"
                raw_pvals.append(p)
                rows_data.append((m1, m2, stat, p, len(sub)))
            except Exception:
                raw_pvals.append(np.nan)
                rows_data.append((m1, m2, np.nan, np.nan, len(sub)))

        valid_mask = [not pd.isna(p) for p in raw_pvals]
        valid_pvals = [p for p, ok in zip(raw_pvals, valid_mask) if ok]
        corrected_pvals = [np.nan] * len(raw_pvals)
        if valid_pvals:
            try:
                _, p_corr, _, _ = multipletests(valid_pvals, alpha=0.05, method='bonferroni' if correction == 'bonferroni' else 'fdr_bh')
                it = iter(p_corr)
                corrected_pvals = [next(it) if ok else np.nan for ok in valid_mask]
            except Exception:
                pass

        method_label = "Bonferroni" if correction == 'bonferroni' else "Benjamini-Hochberg FDR"
        out = f"<table class='apa'><tr><th>Comparison</th><th>Statistic ({'W' if nonparametric else 't'})</th><th>Raw p</th><th>{method_label}-Corrected p</th><th>N</th></tr>"
        for (m1, m2, stat, p, n), p_corr in zip(rows_data, corrected_pvals):
            if pd.isna(p):
                out += f"<tr><td style='text-align:left;'>{m1} vs {m2}</td><td colspan='3' style='color:#9CA3AF;'>Not enough data</td><td>{n}</td></tr>"
                continue
            sig_corr = (not pd.isna(p_corr)) and p_corr < 0.05
            p_corr_str = f"<span class='sig'>{self.apa_p(p_corr)}</span>" if sig_corr else self.apa_p(p_corr)
            out += f"<tr><td style='text-align:left;'>{m1} vs {m2}</td><td>{self.fmt(stat)}</td><td>{self.apa_p(p)}</td><td>{p_corr_str}</td><td>{n}</td></tr>"
        out += "</table>"
        out += (
            f"<p style='font-size:12.5px; color:#6B7280;'><i>{method_label} correction controls the "
            f"family-wise error rate across these {len(pairs)} comparisons. Use the corrected p-value column "
            f"to judge significance, not the raw p-value.</i></p>"
        )
        return out


    def build_raincloud_plot(self, groups_dict, dv_label, title="Raincloud Plot"):
        """
        Builds a hand-rolled raincloud plot (Allen et al., 2019): a half-violin
        kernel-density "cloud" + jittered raw-data "rain" + a slim boxplot, one row
        per group. `groups_dict` is an ordered dict-like of {group_label: 1D array}.
        Returns a QWidget ready to drop into a results tab, or None if there isn't
        enough data to plot (fewer than 2 valid points in any group, or no groups).
        Used for Independent-Samples T-Tests and One-Way ANOVA, as requested.
        """
        if not MATPLOTLIB_AVAILABLE or not groups_dict:
            return None
        from scipy.stats import gaussian_kde

        labels = list(groups_dict.keys())
        cleaned = {}
        for lbl in labels:
            arr = np.asarray(groups_dict[lbl], dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) >= 2 and np.std(arr) > 0:
                cleaned[lbl] = arr
        if not cleaned:
            return None
        labels = list(cleaned.keys())
        n_groups = len(labels)

        is_dark = self.is_dark_mode
        palette = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#06B6D4', '#8B5CF6', '#EC4899', '#84CC16']

        fig = Figure(figsize=(7.5, max(3.5, 1.3 * n_groups + 1.2)))
        ax = fig.add_subplot(111)
        if is_dark:
            fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#1F2937')
        text_color = 'white' if is_dark else 'black'

        for idx, lbl in enumerate(labels):
            vals = cleaned[lbl]
            y_base = idx
            color = palette[idx % len(palette)]

            try:
                kde = gaussian_kde(vals)
                x_grid = np.linspace(vals.min(), vals.max(), 200)
                density = kde(x_grid)
                density = density / density.max() * 0.4
                ax.fill_between(x_grid, y_base + 0.08, y_base + 0.08 + density,
                                 color=color, alpha=0.55, linewidth=0, zorder=2)
            except Exception:
                pass

            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.12, -0.02, size=len(vals))
            ax.scatter(vals, y_base + jitter, s=10, color=color, alpha=0.5, zorder=1, edgecolors='none')

            bp = ax.boxplot([vals], positions=[y_base + 0.03], vert=False, widths=0.06,
                             patch_artist=True, showfliers=False, zorder=3)
            for patch in bp['boxes']:
                patch.set_facecolor(color); patch.set_alpha(0.85)
            for el in ('whiskers', 'caps', 'medians'):
                for line in bp[el]:
                    line.set_color(text_color)

        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(labels, color=text_color)
        ax.set_xlabel(dv_label, color=text_color)
        ax.tick_params(axis='x', colors=text_color)
        ax.set_title(title, fontweight='bold', color=text_color)
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

        chart = self.make_zoomable_chart(fig)
        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)
        note = QLabel("Raincloud plot: distribution shape (cloud), raw observations (rain), and median/IQR (box) shown together.")
        note.setObjectName("InfoLabel")
        note.setWordWrap(True)
        wl.addWidget(note)
        wl.addWidget(chart)
        return wrapper

    # ==========================================
    # GLOBAL SETTINGS & STYLING
    # ==========================================
    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        new_win_action = QAction("New Window (Work on 2nd Dataset)", self)
        new_win_action.triggered.connect(self.open_new_window)
        file_menu.addAction(new_win_action)
        
        file_menu.addSeparator()
        
        open_action = QAction("Open Dataset (CSV, Excel, SPSS)", self)
        open_action.triggered.connect(self.load_data)
        file_menu.addAction(open_action)
        
        sample_menu = file_menu.addMenu("Load Sample Data")
        load_exp = QAction("Experiment Example (N=300) — Contact Study", self)
        load_exp.triggered.connect(lambda: self.load_sample("experiment_example.csv"))
        sample_menu.addAction(load_exp)

        load_surv = QAction("Survey Example (N=250) — Psychometric Survey", self)
        load_surv.triggered.connect(lambda: self.load_sample("survey_example.csv"))
        sample_menu.addAction(load_surv)

        sample_menu.addSeparator()
        guide_action2 = QAction("Examples Guide (what to analyse with each dataset)", self)
        guide_action2.triggered.connect(lambda: self.open_document("guide"))
        sample_menu.addAction(guide_action2)

        save_action = QAction("Save Dataset As...", self)
        save_action.triggered.connect(self.save_dataset)
        file_menu.addAction(save_action)
        
        edit_menu = menubar.addMenu("Edit")
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo_data)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.triggered.connect(self.redo_data)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)

        settings_menu = menubar.addMenu("Settings")
        theme_action = QAction("Toggle Light/Dark Mode", self)
        theme_action.triggered.connect(self.toggle_theme)
        settings_menu.addAction(theme_action)
        
        dec_menu = settings_menu.addMenu("Decimal Formatting")
        for val in [2, 3, 4]:
            act = QAction(f"{val} Decimals", self)
            act.triggered.connect(lambda checked, v=val: self.set_decimals(v))
            dec_menu.addAction(act)
            
        help_menu = menubar.addMenu("Help")

        manual_action = QAction("User Manual", self)
        manual_action.triggered.connect(lambda: self.open_document("manual"))
        help_menu.addAction(manual_action)

        readme_action = QAction("README / About PsyStat", self)
        readme_action.triggered.connect(lambda: self.open_document("readme"))
        help_menu.addAction(readme_action)

        guide_action = QAction("Example Datasets Guide", self)
        guide_action.triggered.connect(lambda: self.open_document("guide"))
        help_menu.addAction(guide_action)

        help_menu.addSeparator()

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def load_sample(self, filename):
        # Use resources.py loader — resolves correctly from both source and bundle
        try:
            from resources import load_example_dataset, get_example_dataset_info
            self.df = load_example_dataset(filename)
            info    = get_example_dataset_info().get(filename, {})
            label   = info.get("label", filename)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load sample dataset:\n{str(e)}")
            return

        # Re-initialize variable metadata
        self.var_labels       = {col: "" for col in self.df.columns}
        self.var_value_labels = {col: "" for col in self.df.columns}
        self.var_scales       = {}
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.var_scales[col] = "Nominal"
            elif self.df[col].dtype in ['float64', 'float32']:
                self.var_scales[col] = "Ratio"
            else:
                self.var_scales[col] = "Interval"

        self.update_global_dropdowns()
        if hasattr(self, 'populate_data_tables'):
            self.populate_data_tables()

        QMessageBox.information(
            self, "Sample Dataset Loaded",
            f"Loaded: {label}\nRows: {len(self.df)} | Columns: {len(self.df.columns)}\n\n"
            "See  File → Load Sample Data → Examples Guide  for suggested analyses."
        )

    def open_document(self, doc_key):
        """
        Open a bundled document in the system's default viewer.

        doc_key can be:
          "manual"  → docs/USER_MANUAL.md   (opens as HTML in browser)
          "readme"  → README.md             (opens as HTML in browser)
          "guide"   → examples/EXAMPLES_GUIDE.md
        """
        try:
            from resources import open_user_manual, resource_path as rp
        except ImportError:
            QMessageBox.critical(self, "Error", "resources.py not found in bundle.")
            return

        if doc_key == "manual":
            open_user_manual()
            return

        # For other docs, resolve path and open with system viewer
        path_map = {
            "readme": "README.md",
            "guide":  "examples/EXAMPLES_GUIDE.md",
        }
        rel = path_map.get(doc_key, doc_key)
        try:
            path = rp(rel)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        import subprocess, platform, webbrowser
        try:
            if platform.system() == 'Darwin':
                subprocess.call(('open', path))
            elif platform.system() == 'Windows':
                os.startfile(path)
            else:
                subprocess.call(('xdg-open', path))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open document: {str(e)}")

    def show_about(self):
        QMessageBox.about(
            self,
            "About PsyStat",
            "<h2>PsyStat v1.1</h2>"
            "<p>A Comprehensive Statistical Analysis Tool<br>"
            "for Psychology &amp; Social Science Research</p>"
            "<p><b>Creator:</b> Tery Setiawan<br>"
            "<b>Affiliations:</b> Universitas Kristen Maranatha &amp; Radboud University</p>"
            "<p>PsyStat is released under the MIT License for academic and research purposes.<br>"
            "Please cite the creator if you use this tool in published empirical research.</p>"
            "<p><a href='https://github.com/terysetn02-source/psystat'>"
            "github.com/terysetn02-source/psystat</a></p>"
        )

    def open_new_window(self):
        new_win = PsyStat()
        new_win.setGeometry(self.x() + 40, self.y() + 40, self.width(), self.height())
        GLOBAL_WINDOWS.append(new_win)
        new_win.show()

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_modern_theme()
        self.refresh_nav_icons()

    def refresh_nav_icons(self):
        """Regenerates the top nav bar's mini-graph icons with theme-appropriate
        accent colors after a dark/light mode switch."""
        if not hasattr(self, 'nav_buttons'):
            return
        for i, btn in enumerate(self.nav_buttons):
            try:
                pixmap = generate_nav_icon(self.nav_items[i], dark=self.is_dark_mode, size_px=48, dpi=100)
                if pixmap is not None and not pixmap.isNull():
                    scaled = pixmap.scaled(32, 32,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                    btn.icon_label.setPixmap(scaled)
            except Exception:
                pass

    def apply_modern_theme(self):
        # Added min-width to input fields so they squash properly instead of forcing layout width
        if self.is_dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #111827; color: #F9FAFB; }
                QListWidget { 
                    background-color: #1E293B; 
                    border-right: 1px solid #334155; 
                    font-family: 'Segoe UI', Arial; font-size: 15px; padding: 5px 0px; 
                }
                QListWidget::item { padding: 10px 15px; border-left: 4px solid transparent; color: #CBD5E1; }
                QListWidget::item:hover { background-color: #334155; }
                QListWidget::item:selected { background-color: #334155; color: #818CF8; border-left: 4px solid #818CF8; font-weight: bold; }
                QWidget#NavBarContainer { background-color: #1E293B; }
                QFrame#NavBarSeparator { background-color: #334155; border: none; }
                QWidget#NavTopButton {
                    background-color: transparent; border: none; border-radius: 8px;
                }
                QWidget#NavTopButton:hover { background-color: #334155; }
                QWidget#NavTopButton[checked="true"] {
                    background-color: #312E81; border-bottom: 3px solid #818CF8;
                }
                QGroupBox { font-weight: bold; font-size: 16px; border: 1px solid #374151; border-radius: 6px; margin-top: 15px; padding-top: 20px; background-color: #1F2937; }
                QLabel { font-size: 14px; font-weight: bold; color: #F9FAFB; }
                QLabel#InfoLabel { background-color: #1E293B; border: 1px solid #334155; color: #94A3B8; padding: 12px; border-radius: 6px; font-size: 13px; font-weight: normal; }
                QPushButton { background-color: #4F46E5; color: white; padding: 10px 15px; border-radius: 6px; font-weight: bold; font-size: 14px; border: none; }
                QPushButton:hover { background-color: #4338CA; }
                QPushButton:checked { background-color: #1E3A8A; border: 2px inset #1E40AF; }
                QPushButton#AdvBtn { background-color: #059669; padding: 4px 10px; font-size: 12px; }
                QPushButton#AdvBtn:hover { background-color: #047857; }
                QPushButton#ClearBtn { background-color: #4B5563; padding: 4px 8px; font-size: 12px; }
                QPushButton#ClearBtn:hover { background-color: #374151; }
                QSpinBox, QDoubleSpinBox, QLineEdit { min-width: 50px; padding: 9px 10px; border: 1.5px solid #4B5563; border-radius: 6px; background-color: #374151; color: white; font-size: 15px; }
                QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border: 1.5px solid #818CF8; background-color: #3F4A5E; }
                QComboBox {
                    min-width: 50px; padding: 9px 14px; border: 1.5px solid #4B5563; border-radius: 6px;
                    background-color: #374151; color: #F9FAFB; font-size: 15px; font-weight: 500;
                }
                QComboBox:hover { border: 1.5px solid #6366F1; background-color: #3F4A5E; }
                QComboBox:focus { border: 1.5px solid #818CF8; }
                QComboBox:on { border: 1.5px solid #818CF8; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; }
                QComboBox::drop-down { border: none; width: 30px; }
                QComboBox::down-arrow {
                    width: 0px; height: 0px;
                    border-left: 5px solid transparent; border-right: 5px solid transparent;
                    border-top: 6px solid #9CA3AF; margin-right: 10px;
                }
                QComboBox:hover::down-arrow { border-top: 6px solid #C7D2FE; }
                QComboBox QAbstractItemView {
                    background-color: #2A3344; color: #F1F5F9; border: 1.5px solid #818CF8;
                    border-top: none; outline: none; padding: 4px; font-size: 14.5px;
                    selection-background-color: #4F46E5; selection-color: white;
                }
                QTabWidget::pane { border: 1px solid #374151; background: #1F2937; }
                QTabBar::tab { background: #374151; border: 1px solid #4B5563; padding: 10px 20px; color: #D1D5DB; font-weight: bold;}
                QTabBar::tab:selected { background: #1F2937; color: #818CF8; border-bottom: none; }
                QTableWidget { background-color: #1F2937; color: #F9FAFB; gridline-color: #374151; }
                QHeaderView::section { background-color: #374151; color: white; border: 1px solid #4B5563; }
                QTextEdit { background-color: #1F2937; color: white; border: 1px solid #374151; }
                QGroupBox#VarBankBox { border: 2px dashed #818CF8; background-color: #1E2A45; }
                QListWidget#VarBank { background-color: #1E2A45; border: 1px solid #4338CA; border-radius: 4px; }
                QListWidget#VarBank::item { border-left: 3px solid #818CF8; margin: 2px 4px; border-radius: 3px; background-color: #2A3A5C; }
                QListWidget#VarBank::item:hover { background-color: #374B7A; }
                QListWidget#DropTarget { border: 1px dashed #6366F1; }
                QWidget#BankPanel { background-color: #1E2A45; border-right: 2px solid #4338CA; }
                QLabel#BankTitle { font-size: 14px; font-weight: bold; color: #C7D2FE; background: transparent; border: none; padding: 2px 0px; }
                QLabel#DropHint { font-size: 11px; font-style: italic; color: #6B7280; background: transparent; border: none; padding: 1px 2px; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #F3F4F6; }
                QListWidget { 
                    background-color: #F8FAFC; 
                    border-right: 1px solid #E2E8F0; 
                    font-family: 'Segoe UI', Arial; font-size: 15px; padding: 5px 0px; 
                }
                QListWidget::item { padding: 10px 15px; border-left: 4px solid transparent; color: #334155; }
                QListWidget::item:hover { background-color: #E2E8F0; }
                QListWidget::item:selected { background-color: #E2E8F0; color: #4F46E5; border-left: 4px solid #4F46E5; font-weight: bold; }
                QWidget#NavBarContainer { background-color: #F8FAFC; }
                QFrame#NavBarSeparator { background-color: #E2E8F0; border: none; }
                QWidget#NavTopButton {
                    background-color: transparent; border: none; border-radius: 8px;
                }
                QWidget#NavTopButton:hover { background-color: #E2E8F0; }
                QWidget#NavTopButton[checked="true"] {
                    background-color: #EEF2FF; border-bottom: 3px solid #4F46E5;
                }
                QGroupBox { font-weight: bold; font-size: 16px; border: 1px solid #D1D5DB; border-radius: 6px; margin-top: 15px; padding-top: 20px; background-color: #FFFFFF; }
                QLabel { font-size: 14px; font-weight: bold; color: #111827; }
                QLabel#InfoLabel { background-color: #F0Fdf4; border: 1px solid #BBF7D0; color: #166534; padding: 12px; border-radius: 6px; font-size: 13px; font-weight: normal;}
                QPushButton { background-color: #4F46E5; color: white; padding: 10px 15px; border-radius: 6px; font-weight: bold; font-size: 14px; border: none;}
                QPushButton:hover { background-color: #4338CA; }
                QPushButton:checked { background-color: #1E3A8A; border: 2px inset #1E40AF; }
                QPushButton#AdvBtn { background-color: #10B981; padding: 4px 10px; font-size: 12px; }
                QPushButton#AdvBtn:hover { background-color: #059669; }
                QPushButton#ClearBtn { background-color: #9CA3AF; padding: 4px 8px; font-size: 12px; }
                QPushButton#ClearBtn:hover { background-color: #6B7280; }
                QSpinBox, QDoubleSpinBox, QLineEdit { min-width: 50px; padding: 9px 10px; border: 1.5px solid #D1D5DB; border-radius: 6px; background-color: white; font-size: 15px; color: #111827; }
                QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border: 1.5px solid #6366F1; background-color: #FAFAFF; }
                QComboBox {
                    min-width: 50px; padding: 9px 14px; border: 1.5px solid #D1D5DB; border-radius: 6px;
                    background-color: white; color: #111827; font-size: 15px; font-weight: 500;
                }
                QComboBox:hover { border: 1.5px solid #818CF8; background-color: #FAFAFF; }
                QComboBox:focus { border: 1.5px solid #4F46E5; }
                QComboBox:on { border: 1.5px solid #4F46E5; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; }
                QComboBox::drop-down { border: none; width: 30px; }
                QComboBox::down-arrow {
                    width: 0px; height: 0px;
                    border-left: 5px solid transparent; border-right: 5px solid transparent;
                    border-top: 6px solid #6B7280; margin-right: 10px;
                }
                QComboBox:hover::down-arrow { border-top: 6px solid #4338CA; }
                QComboBox QAbstractItemView {
                    background-color: white; color: #111827; border: 1.5px solid #4F46E5;
                    border-top: none; outline: none; padding: 4px; font-size: 14.5px;
                    selection-background-color: #4F46E5; selection-color: white;
                }
                QTabWidget::pane { border: 1px solid #D1D5DB; background: white; }
                QTabBar::tab { background: #F3F4F6; border: 1px solid #D1D5DB; padding: 10px 20px; color: #4B5563; font-weight: bold;}
                QTabBar::tab:selected { background: white; color: #4F46E5; border-bottom: none; }
                QTableWidget { background-color: white; color: black; }
                QTextEdit { background-color: white; color: black; border: 1px solid #D1D5DB; }
                QGroupBox#VarBankBox { border: 2px dashed #818CF8; background-color: #EEF2FF; }
                QListWidget#VarBank { background-color: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 4px; }
                QListWidget#VarBank::item { border-left: 3px solid #6366F1; margin: 2px 4px; border-radius: 3px; background-color: #FFFFFF; }
                QListWidget#VarBank::item:hover { background-color: #E0E7FF; }
                QListWidget#DropTarget { border: 1px dashed #818CF8; }
                QWidget#BankPanel { background-color: #EEF2FF; border-right: 2px solid #C7D2FE; }
                QLabel#BankTitle { font-size: 14px; font-weight: bold; color: #4338CA; background: transparent; border: none; padding: 2px 0px; }
                QLabel#DropHint { font-size: 11px; font-style: italic; color: #9CA3AF; background: transparent; border: none; padding: 1px 2px; }
            """)

    def get_apa_css(self):
        bg = "#1F2937" if self.is_dark_mode else "#FFFFFF"
        text = "#F9FAFB" if self.is_dark_mode else "#111827"
        border = "#4B5563" if self.is_dark_mode else "#E5E7EB"
        th_bg = "#374151" if self.is_dark_mode else "#F9FAFB"
        box_border = "#818CF8" if self.is_dark_mode else "#4F46E5"
        
        pre_bg = "#111827" if self.is_dark_mode else "#F8FAFC"
        pre_text = "#E2E8F0" if self.is_dark_mode else "#1E293B"
        pre_border = "#374151" if self.is_dark_mode else "#E2E8F0"
        
        return f"""
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; color: {text}; background-color: {bg}; line-height: 1.6; font-size: 15px; padding: 10px; }}
            h2 {{ color: {box_border}; border-bottom: 2px solid {border}; padding-bottom: 5px; margin-top: 20px; font-size: 20px; }}
            h3 {{ font-size: 16px; margin-top: 15px; margin-bottom: 5px; }}
            table.apa {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 15px; color: {text}; }}
            table.apa th, table.apa td {{ padding: 10px 15px; text-align: center; border-bottom: 1px solid {border}; }}
            table.apa th {{ border-top: 2px solid {text}; border-bottom: 2px solid {text}; font-weight: bold; background-color:{th_bg}; }}
            table.apa tr:last-child td {{ border-bottom: 2px solid {text}; }}
            
            .interpret {{ margin-top: 5px; font-size: 13px; font-style: italic; color: #6B7280; padding: 0; line-height: 1.4; }}
            .interpret b {{ font-style: normal; color: {text}; }}
            
            .warn {{ color: #EF4444; font-weight: bold; }}
            .sig {{ font-weight: bold; color: #10B981; }}
            
            pre.lavaan {{ background-color: {pre_bg}; color: {pre_text}; padding: 15px; border-radius: 5px; border: 1px solid {pre_border}; font-family: Consolas, monospace; font-size: 13px; white-space: pre-wrap; }}
        </style>
        """

    def display_tab(self, index):
        self.tabs.setCurrentIndex(index)
        if hasattr(self, 'nav_buttons') and 0 <= index < len(self.nav_buttons):
            for i, btn in enumerate(self.nav_buttons):
                btn.setChecked(i == index)

    def fmt(self, val, is_p=False):
        if pd.isna(val) or val is None or val == '-': return "-"
        if is_p and isinstance(val, (int, float)) and val < 0.001: return "&lt; .001"
        try: return f"{float(val):.{self.decimals}f}"
        except: return str(val)

    def apa_p(self, p):
        """APA-style p-value string for inline prose, e.g. 'p < .001' or 'p = .032'."""
        try:
            p = float(p)
        except (TypeError, ValueError):
            return "p = n/a"
        if pd.isna(p): return "p = n/a"
        if p < 0.001: return "p &lt; .001"
        return f"p = {self.fmt(p, True)}"

    def build_apa_writeup(self, kind, d):
        """
        Generates a distinct green 'APA Journal Write-Up' block: ready-to-paste prose
        following APA 7th-edition reporting conventions. `kind` selects the template;
        `d` is a dict of the values that template needs (each generator below documents
        what it expects). Unknown kinds or missing keys degrade gracefully -- this never
        raises, so a write-up failure can't break the surrounding analysis output.
        """
        try:
            text = self._apa_writeup_text(kind, d)
        except Exception:
            text = None
        if not text:
            return ""
        return (
            "<div style='margin-top:18px; background:#ECFDF5; border:2px solid #10B981; "
            "border-radius:10px; padding:16px 18px;'>"
            "<div style='font-weight:900; color:#047857; font-size:13px; letter-spacing:0.5px; "
            "margin-bottom:8px;'>📄 APA JOURNAL WRITE-UP</div>"
            f"<div style='color:#064E3B; font-size:14.5px; line-height:1.7;'>{text}</div>"
            "</div>"
        )

    def _apa_writeup_text(self, kind, d):
        f, p = self.fmt, self.apa_p

        if kind == "Descriptives":
            # d: {var, n, mean, sd, skew, kurt, shapiro_p}
            txt = (f"For the variable <i>{d['var']}</i>, the mean was M = {f(d['mean'])} "
                   f"(SD = {f(d['sd'])}, N = {d['n']}).")
            if d.get('shapiro_p') is not None:
                normal = "did not significantly deviate" if (pd.notna(d['shapiro_p']) and d['shapiro_p'] >= 0.05) else "significantly deviated"
                txt += (f" A Shapiro-Wilk test indicated the distribution {normal} from normality, "
                        f"{p(d['shapiro_p'])}.")
            return txt

        if kind == "Correlation":
            # d: {method, var1, var2, r, p, n}
            direction = "positive" if d['r'] >= 0 else "negative"
            strength = "negligible" if abs(d['r']) < 0.10 else "small" if abs(d['r']) < 0.30 else "medium" if abs(d['r']) < 0.50 else "large"
            sig = "a statistically significant" if (pd.notna(d['p']) and d['p'] < 0.05) else "no statistically significant"
            return (f"A {d['method']} correlation revealed {sig} {strength} {direction} relationship between "
                    f"<i>{d['var1']}</i> and <i>{d['var2']}</i>, r({d['n']-2}) = {f(d['r'])}, {p(d['p'])}.")

        if kind == "Reliability":
            # d: {k, alpha, omega}
            txt = f"The {d['k']}-item scale demonstrated "
            parts = []
            if pd.notna(d.get('alpha')):
                parts.append(f"acceptable internal consistency (Cronbach's α = {f(d['alpha'])})" if d['alpha'] >= 0.70 else f"internal consistency below conventional thresholds (Cronbach's α = {f(d['alpha'])})")
            if pd.notna(d.get('omega')):
                parts.append(f"McDonald's ω = {f(d['omega'])}")
            txt += " and ".join(parts) + "."
            return txt

        if kind == "TTest":
            # d: {dv, iv, design, stat_name, stat, df_val, p, d_val, g1_name, g1_mean, g1_sd, g2_name, g2_mean, g2_sd, welch_used (optional)}
            sig = "a statistically significant" if (pd.notna(d['p']) and d['p'] < 0.05) else "no statistically significant"
            df_part = f"({f(d['df_val'])})" if d.get('df_val') is not None else ""
            welch_phrase = " (using Welch's correction for unequal variances)" if d.get('welch_used') else ""
            return (f"An {d['design']} t-test{welch_phrase} showed {sig} difference in <i>{d['dv']}</i> between "
                    f"{d['g1_name']} (M = {f(d['g1_mean'])}, SD = {f(d['g1_sd'])}) and {d['g2_name']} "
                    f"(M = {f(d['g2_mean'])}, SD = {f(d['g2_sd'])}), {d['stat_name']}{df_part} = {f(d['stat'])}, "
                    f"{p(d['p'])}, d = {f(d['d_val'])}.")

        if kind == "ANOVA":
            # d: {dv, iv_term, f_val, df1, df2, p, eta2}
            sig = "a statistically significant" if (pd.notna(d['p']) and d['p'] < 0.05) else "no statistically significant"
            return (f"A one-way ANOVA showed {sig} effect of <i>{d['iv_term']}</i> on <i>{d['dv']}</i>, "
                    f"F({d['df1']}, {d['df2']}) = {f(d['f_val'])}, {p(d['p'])}, partial η² = {f(d['eta2'])}.")

        if kind == "Regression":
            # d: {dv, r2, f_val, df1, df2, p}
            sig = "significantly predicted" if (pd.notna(d['p']) and d['p'] < 0.05) else "did not significantly predict"
            return (f"The regression model {sig} <i>{d['dv']}</i>, R² = {f(d['r2'])}, "
                    f"F({d['df1']}, {d['df2']}) = {f(d['f_val'])}, {p(d['p'])}.")

        if kind == "Mediation":
            # d: {x, m, y, a, a_p, b, b_p, c, c_p, cprime, cprime_p, indirect, ci_lo, ci_hi}
            mediation_type = "full" if (pd.notna(d['cprime_p']) and d['cprime_p'] >= 0.05 and pd.notna(d['c_p']) and d['c_p'] < 0.05) else "partial" if (pd.notna(d['cprime_p']) and d['cprime_p'] < 0.05) else "no"
            return (f"As predicted, <i>{d['x']}</i> significantly predicted <i>{d['m']}</i>, a = {f(d['a'])}, {p(d['a_p'])}, "
                    f"and <i>{d['m']}</i> significantly predicted <i>{d['y']}</i> controlling for <i>{d['x']}</i>, "
                    f"b = {f(d['b'])}, {p(d['b_p'])}. The indirect effect of <i>{d['x']}</i> on <i>{d['y']}</i> through "
                    f"<i>{d['m']}</i> was ab = {f(d['indirect'])}, 95% CI [{f(d['ci_lo'])}, {f(d['ci_hi'])}]"
                    f"{' (excludes zero, indicating a significant indirect effect)' if (d['ci_lo'] > 0 or d['ci_hi'] < 0) else ' (includes zero, indicating a non-significant indirect effect)'}. "
                    f"The direct effect controlling for the mediator was c' = {f(d['cprime'])}, {p(d['cprime_p'])}, "
                    f"consistent with {mediation_type} mediation.")

        return None

    def set_decimals(self, val):
        self.decimals = val

    def add_info_box(self, layout, text):
        info = QLabel(text)
        info.setObjectName("InfoLabel")
        info.setWordWrap(True)
        info.setMinimumWidth(10) # FLUID WRAP TRICK
        layout.addWidget(info)

    def build_missing_data_note(self, original_df, var_list, clean_n):
        """
        Generates a small transparency note reporting how many rows were dropped
        via listwise deletion before an analysis ran, plus a per-variable missing
        count so the user can see whether missingness is concentrated in one
        variable (a flag for non-random missingness) rather than spread evenly.
        Returns an empty string if there was no missing data (nothing to report),
        so it never adds visual noise to a complete dataset's output.
        `var_list` should be the exact columns used as analysis input (before
        .dropna() was applied) -- pass original_df[var_list] for the comparison.
        """
        try:
            total_n = len(original_df)
            if total_n == 0 or clean_n == total_n:
                return ""
            n_dropped = total_n - clean_n
            pct_dropped = (n_dropped / total_n) * 100

            per_var_html = ""
            if len(var_list) > 1:
                miss_counts = original_df[var_list].isna().sum()
                miss_counts = miss_counts[miss_counts > 0].sort_values(ascending=False)
                if len(miss_counts) > 0:
                    rows = "".join(
                        f"<tr><td style='text-align:left;'>{v}</td><td>{int(c)}</td><td>{c/total_n*100:.1f}%</td></tr>"
                        for v, c in miss_counts.items()
                    )
                    per_var_html = (
                        "<table class='apa' style='margin-top:8px; font-size:12.5px;'>"
                        "<tr><th>Variable</th><th>Missing (n)</th><th>Missing (%)</th></tr>"
                        f"{rows}</table>"
                    )

            severity_color = "#D97706" if pct_dropped >= 10 else "#6B7280"
            severity_note = (
                " This is a substantial proportion — consider whether the data are missing completely at "
                "random (MCAR); if not, listwise deletion (used here) can bias estimates, and an imputation "
                "method may be more appropriate."
                if pct_dropped >= 10 else ""
            )

            return (
                f"<div style='background:#FFFBEB; border-left:4px solid {severity_color}; "
                f"padding:8px 12px; margin:10px 0; font-size:13px; color:#78350F;'>"
                f"⚠ <b>Listwise deletion:</b> {n_dropped} of {total_n} rows ({pct_dropped:.1f}%) were excluded "
                f"due to missing data on the variables used in this analysis (N analyzed = {clean_n})."
                f"{severity_note}{per_var_html}</div>"
            )
        except Exception:
            return ""

    def build_sample_size_warning(self, n, test_kind):
        """
        Generates a small advisory (not blocking) note when N falls below commonly-
        cited minimums for stable estimation of a given test. These are heuristic
        floors from the methods literature, not hard statistical requirements --
        the analysis still runs and is still valid to report; the note exists so
        published-paper reviewers' first question ("was this adequately powered?")
        is answered proactively rather than discovered after submission. Returns ""
        when N is adequate, so it never adds noise to well-powered analyses.
        """
        thresholds = {
            "ttest":       (30, "Independent/Paired t-tests are commonly recommended to have N ≥ 30 per group for the Central Limit Theorem to support approximate normality of the sampling distribution (Field, 2013)."),
            "anova":       (20, "One-way ANOVA designs are commonly recommended to have N ≥ 20 per group to keep Type I error rates stable under mild violations of normality (Field, 2013)."),
            "correlation": (30, "Correlation coefficients are commonly recommended to have N ≥ 30 for the sampling distribution of r to approximate normality and for the estimate to stabilize (Schönbrodt & Perugini, 2013)."),
            "regression":  (None, "Regression models are commonly recommended to have at least N ≥ 10–20 observations per predictor to avoid overfitting and unstable coefficient estimates (Green, 1991)."),
            "mediation":   (100, "Bootstrapped mediation analyses are commonly recommended to have N ≥ 100 for the indirect-effect confidence interval to be reasonably precise (Fritz & MacKinnon, 2007)."),
            "factor":      (None, "Factor analyses are commonly recommended to have N ≥ 5–10 observations per item, or N ≥ 200 in absolute terms, for stable factor loadings (Comrey & Lee, 1992)."),
        }
        if test_kind not in thresholds:
            return ""
        min_n, citation = thresholds[test_kind]
        if min_n is None or n >= min_n:
            return ""
        return (
            f"<div style='background:#FEF2F2; border-left:4px solid #EF4444; padding:8px 12px; "
            f"margin:10px 0; font-size:13px; color:#7F1D1D;'>"
            f"⚠ <b>Small sample size:</b> N = {n} is below the commonly recommended minimum of {min_n} for this test. "
            f"{citation} Results are still reported below, but treat p-values and effect sizes with caution, "
            f"and consider running an a-priori power analysis in the <b>Power Analysis</b> tab before collecting more data.</div>"
        )

    def open_var_dialog(self, list_widget, title):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return
            
        if list_widget in [self.anova_iv, self.dummy_var, self.catpca_vars, self.xtab_v1, self.xtab_v2, self.desc_list]:
            available = self.df.columns.tolist()
        else:
            available = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        selected = [item.text() for item in list_widget.selectedItems()]
        
        dlg = VarSelectDialog(available, selected, title, self)
        if dlg.exec():
            new_selected = dlg.get_selected()
            list_widget.clearSelection()
            for i in range(list_widget.count()):
                it = list_widget.item(i)
                is_sel = it.text() in new_selected
                it.setSelected(is_sel)
                it.setHidden(not is_sel)

    def clear_drop_target(self, list_widget):
        """Clears selection AND re-hides every row, returning a drop-target list to
        its visually-empty default state (used by each box's 'Clear' button)."""
        list_widget.clearSelection()
        for i in range(list_widget.count()):
            list_widget.item(i).setHidden(True)

    def enable_drag_drop_target(self, list_widget, single=False):
        """
        Upgrades a plain QListWidget into a JASP-style drop target IN PLACE, without
        requiring the widget to be constructed as a DropListWidget subclass. This keeps
        every existing `self.xxx_vars.selectedItems()` read in every analysis function
        working untouched -- dropping a variable from the Available Variables bank simply
        selects the matching row in this list (or, if `single` is True, replaces the prior
        single selection -- the auto-swap behavior for one-slot boxes like a DV picker).
        """
        list_widget.setAcceptDrops(True)
        list_widget._psystat_single_drop = single

        def _drag_enter(event, w=list_widget):
            if event.mimeData().hasFormat(VAR_MIME):
                event.acceptProposedAction()
            else:
                event.ignore()

        def _drag_move(event, w=list_widget):
            if event.mimeData().hasFormat(VAR_MIME):
                event.acceptProposedAction()
            else:
                event.ignore()

        def _drop(event, w=list_widget):
            if not event.mimeData().hasFormat(VAR_MIME):
                event.ignore()
                return
            names = [n.strip() for n in bytes(event.mimeData().data(VAR_MIME)).decode("utf-8").split(",") if n.strip()]
            if not names:
                event.ignore()
                return
            if getattr(w, '_psystat_single_drop', False):
                # Auto-swap: the previously selected item goes back to looking empty
                # (re-hidden) since only one slot is allowed here.
                for i in range(w.count()):
                    prev = w.item(i)
                    if prev.isSelected():
                        prev.setSelected(False)
                        prev.setHidden(True)
                names = names[:1]
            matched = False
            unmatched = []
            for name in names:
                hit = False
                for i in range(w.count()):
                    it = w.item(i)
                    if it.text() == name:
                        it.setSelected(True)
                        it.setHidden(False)
                        matched = True
                        hit = True
                        break
                if not hit:
                    unmatched.append(name)
            event.acceptProposedAction()
            if unmatched:
                self.statusBar().showMessage(
                    f"⚠ '{', '.join(unmatched)}' not valid for this box (wrong type or already placed elsewhere).", 4000
                )
            if matched:
                self.statusBar().showMessage("✓ Variable added.", 1500)

        list_widget.dragEnterEvent = _drag_enter
        list_widget.dragMoveEvent = _drag_move
        list_widget.dropEvent = _drop
        list_widget.setToolTip("Drag variables here from the 'Available Variables' bank, or use Pop-up Select.")

    def enable_drag_drop_combo(self, combo):
        """
        Upgrades a plain QComboBox into a drag-and-drop target IN PLACE. Used for
        single-variable boxes (Dependent Variable, X-axis, etc). Dropping a variable
        on it sets currentText -- if a value was already selected, the drop silently
        SWAPS it out (the requested JASP auto-swap-back-to-bank behavior). Drops are
        only accepted if the dragged name is one of the combo's current choices, so a
        numeric-only DV box will correctly reject a categorical/text variable.
        """
        combo.setAcceptDrops(True)

        def _drag_enter(event, w=combo):
            if event.mimeData().hasFormat(VAR_MIME):
                event.acceptProposedAction()
            else:
                event.ignore()

        def _drag_move(event, w=combo):
            if event.mimeData().hasFormat(VAR_MIME):
                event.acceptProposedAction()
            else:
                event.ignore()

        def _drop(event, w=combo):
            if not event.mimeData().hasFormat(VAR_MIME):
                event.ignore()
                return
            names = [n.strip() for n in bytes(event.mimeData().data(VAR_MIME)).decode("utf-8").split(",") if n.strip()]
            if not names:
                event.ignore()
                return
            name = names[0]
            idx = w.findText(name)
            if idx >= 0:
                w.setCurrentIndex(idx)
                event.acceptProposedAction()
                self.statusBar().showMessage(f"✓ '{name}' set.", 1500)
            else:
                event.ignore()
                self.statusBar().showMessage(f"⚠ '{name}' not valid for this box (wrong type).", 4000)

        combo.dragEnterEvent = _drag_enter
        combo.dragMoveEvent = _drag_move
        combo.dropEvent = _drop
        combo.setToolTip("Drag a variable here to set it (replaces current selection).")

    def make_variable_bank(self, numeric_only=False, height=170):
        """
        Creates a compact 'Available Variables' source panel as its own QGroupBox.
        Kept for any caller that still wants a short, stacked bank block. For the
        standard tab layout, prefer build_bank_panel(), which returns a full-height
        panel meant to sit in its own dedicated left column (see create_split_module's
        `bank` parameter) rather than stacked above the input fields.
        """
        box = QGroupBox("📦 Available Variables")
        box.setObjectName("VarBankBox")
        v = QVBoxLayout(box)
        v.setSpacing(4)
        hint = QLabel("Drag a variable into a box on the right →")
        hint.setObjectName("InfoLabel")
        hint.setWordWrap(True)
        v.addWidget(hint)
        bank = VariableBank()
        bank.setMinimumHeight(height)
        bank._numeric_only = numeric_only
        v.addWidget(bank)
        if not hasattr(self, 'var_banks'):
            self.var_banks = []
        self.var_banks.append(bank)
        return box

    def build_bank_panel(self, numeric_only=False):
        """
        Creates the full-height 'Available Variables' panel used as the dedicated
        LEFT-most column of an analysis tab (Bank | Controls | Results), instead of
        being stacked above the controls. This is the standard, uniform pattern: call
        this once per tab and pass the result as create_split_module's `bank=`
        argument. The panel fills all available vertical space so the variable list
        is easy to scan and drag from, with a persistent search box and a clear
        explanation of how to use it (drag to a field on the right, or use the
        Pop-up Select button that remains on every field as a non-drag alternative).
        """
        panel = QWidget()
        panel.setObjectName("BankPanel")
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        title = QLabel("📦 Available Variables")
        title.setObjectName("BankTitle")
        v.addWidget(title)

        hint = QLabel("Drag a variable onto a field to the right, or select a field's \"Pop-up Select\" button instead.")
        hint.setObjectName("InfoLabel")
        hint.setWordWrap(True)
        v.addWidget(hint)

        search = QLineEdit()
        search.setPlaceholderText("🔍 Filter variables…")
        v.addWidget(search)

        bank = VariableBank()
        bank._numeric_only = numeric_only
        bank.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        def filter_bank(text):
            text = text.lower().strip()
            for i in range(bank.count()):
                item = bank.item(i)
                item.setHidden(text not in item.text().lower())
        search.textChanged.connect(filter_bank)

        v.addWidget(bank)
        if not hasattr(self, 'var_banks'):
            self.var_banks = []
        self.var_banks.append(bank)
        panel.setMinimumWidth(190)
        panel.setMaximumWidth(260)
        return panel

    def build_reference_panel(self, title, html_content):
        """
        Creates a left-column reference/legend panel matching the visual style of
        build_bank_panel(), for tabs that have no dataset variables to drag (e.g.
        Power Analysis, which works entirely from manually-entered numbers). Keeps
        the same Bank | Controls | Results three-pane visual rhythm used everywhere
        else in the app, but shows static reference content instead of an empty
        variable list, which would otherwise be confusing.
        """
        panel = QWidget()
        panel.setObjectName("BankPanel")
        v = QVBoxLayout(panel)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("BankTitle")
        title_lbl.setWordWrap(True)
        v.addWidget(title_lbl)

        content = QLabel(html_content)
        content.setObjectName("InfoLabel")
        content.setWordWrap(True)
        content.setTextFormat(Qt.TextFormat.RichText)
        content.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        v.addWidget(scroll)

        panel.setMinimumWidth(190)
        panel.setMaximumWidth(260)
        return panel

    def setup_list_selection(self, layout, label_text, list_widget, dialog_title, single=False):
        lbl = QLabel(label_text)
        lbl.setStyleSheet("margin-top: 5px; margin-bottom: 0px;")
        lbl.setWordWrap(True)
        lbl.setMinimumWidth(10) # FLUID WRAP TRICK
        layout.addWidget(lbl)
        
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(5)
        
        btn_adv = QPushButton("🔍 Pop-up Select")
        btn_adv.setObjectName("AdvBtn")
        btn_adv.clicked.connect(lambda: self.open_var_dialog(list_widget, dialog_title))
        
        btn_clr = QPushButton("Clear")
        btn_clr.setObjectName("ClearBtn")
        btn_clr.clicked.connect(lambda: self.clear_drop_target(list_widget))
        
        h.addWidget(btn_adv)
        h.addWidget(btn_clr)
        h.addStretch()
        
        layout.addLayout(h)
        layout.addWidget(list_widget)

        hint = QLabel("Drag a variable here, or use Pop-up Select above.")
        hint.setObjectName("DropHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Make this box a native drag-and-drop target (JASP-style), in addition to
        # the existing Pop-up Select / Clear buttons which remain fully functional.
        self.enable_drag_drop_target(list_widget, single=single)

    # ==========================================
    # DATA STATE MANAGEMENT
    # ==========================================
    def save_state(self):
        if self.df is not None:
            self.history.append(self.df.copy())
            # Cap undo history so large datasets with long editing sessions don't
            # silently accumulate unbounded memory (each entry is a full DataFrame copy).
            if len(self.history) > self.MAX_UNDO_HISTORY:
                self.history.pop(0)
            self.redo_stack.clear()
            self.undo_action.setEnabled(True)
            self.redo_action.setEnabled(False)

    def undo_data(self):
        if self.history:
            self.redo_stack.append(self.df.copy())
            if len(self.redo_stack) > self.MAX_UNDO_HISTORY:
                self.redo_stack.pop(0)
            self.df = self.history.pop()
            self.undo_action.setEnabled(len(self.history) > 0)
            self.redo_action.setEnabled(True)
            self.update_global_dropdowns()
            if hasattr(self, 'populate_data_tables'): self.populate_data_tables()

    def redo_data(self):
        if self.redo_stack:
            self.history.append(self.df.copy())
            if len(self.history) > self.MAX_UNDO_HISTORY:
                self.history.pop(0)
            self.df = self.redo_stack.pop()
            self.undo_action.setEnabled(True)
            self.redo_action.setEnabled(len(self.redo_stack) > 0)
            self.update_global_dropdowns()
            if hasattr(self, 'populate_data_tables'): self.populate_data_tables()

    # ==========================================
    # HELPER: Split Screen Layout
    # ==========================================
    def make_resizable_canvas(self, fig):
        """
        Wraps a matplotlib Figure in a FigureCanvas that actually grows and shrinks
        with its container. Plain FigureCanvas() does not reliably expand inside
        nested Qt layouts (QSplitter -> QTabWidget -> QVBoxLayout, as used throughout
        this app) -- it sits at its initial figsize/dpi pixel size and clips or
        leaves blank space when the window is resized, unless given an explicit
        Expanding size policy. This is the standard fix used across PyQt+matplotlib
        integrations. Every chart in the app should be created via this helper
        instead of calling FigureCanvas(fig) directly.

        Returns the raw FigureCanvasQTAgg (not a wrapper widget) -- some callers
        (e.g. the interactive draggable SNA network plot) need direct access to
        canvas.mpl_connect()/draw_idle() and would break if this returned something
        else. For a canvas with zoom/pan controls attached, wrap the result in
        make_zoomable_chart() instead of embedding it directly.
        """
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas.updateGeometry()
        return canvas

    def make_zoomable_chart(self, fig):
        """
        Builds a complete chart widget: a resizable canvas (via make_resizable_canvas)
        plus matplotlib's standard navigation toolbar (pan, zoom-to-rectangle, home/
        back/forward, and save-as-image) docked compactly above it. This is the
        standard way to add zoom to every visualization in the app -- use this
        instead of make_resizable_canvas() directly wherever the chart is just being
        displayed (not driving custom mouse-event interactivity like the SNA network
        plot, which needs the bare canvas from make_resizable_canvas() instead).
        Returns a single QWidget ready to addTab()/addWidget() anywhere a canvas was
        used before.
        """
        canvas = self.make_resizable_canvas(fig)
        wrapper = QWidget()
        wrapper.setObjectName("ZoomableChart")
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        toolbar = NavigationToolbar(canvas, wrapper)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setMaximumHeight(32)
        # Remove the live x/y coordinate readout label that NavigationToolbar2QT
        # appends on the right -- it's a small monospace QLabel with no name, so
        # detect it by type/lack of a defaultAction rather than relying on an index
        # that can vary by Qt/matplotlib version.
        for child in toolbar.findChildren(QLabel):
            child.setStyleSheet("font-size: 11px;")

        v.addWidget(toolbar)
        v.addWidget(canvas)
        return wrapper

    def create_split_module(self, controls_widget, bank=None):
        module_tab = QWidget()
        layout = QVBoxLayout(module_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)

        bank_pane_present = bank is not None
        if bank_pane_present:
            splitter.addWidget(bank)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Horizontal bar OFF, forces all inner layouts to squash and wrap
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(controls_widget)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        splitter.addWidget(scroll)
        
        results_panel = QWidget()
        rp_layout = QVBoxLayout(results_panel)
        rp_layout.setContentsMargins(10, 10, 10, 10)
        
        toolbar = QHBoxLayout()
        toolbar.addStretch()
        btn_export = QPushButton("💾 Export to HTML/Word")
        btn_export.setStyleSheet("background-color: #10B981; color: white;")
        toolbar.addWidget(btn_export)
        rp_layout.addLayout(toolbar)
        
        out_tabs = QTabWidget()
        out_tabs.setTabsClosable(True)
        
        def close_tab(index):
            if out_tabs.tabText(index) not in ["Interactive Builder", "Syntax Editor"]:
                out_tabs.removeTab(index)
        out_tabs.tabCloseRequested.connect(close_tab)
        
        rp_layout.addWidget(out_tabs)
        
        def export_results():
            if out_tabs.count() == 0:
                return QMessageBox.warning(self, "Export Empty", "No results to export.")
            path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "HTML Document (*.html);;Word Document (*.doc)")
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(f"<html><head><meta charset='utf-8'>{self.get_apa_css()}</head><body>")
                        for i in range(out_tabs.count()):
                            f.write(f"<h1 style='color:#4F46E5;'>{out_tabs.tabText(i)}</h1>")
                            widget = out_tabs.widget(i)
                            if isinstance(widget, QTextEdit):
                                f.write(widget.toHtml())
                            f.write("<br><hr><br>")
                        f.write("</body></html>")
                    QMessageBox.information(self, "Success", f"Results exported to:\n{path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))
        
        btn_export.clicked.connect(export_results)
        
        splitter.addWidget(results_panel)
        # Explicit stretch factors: side panels (bank, controls) stay narrow and
        # fixed-feeling, while the results column always absorbs almost all of the
        # extra space when the window is resized -- without this, QSplitter's
        # default equal-stretch behavior can leave the results panel not growing/
        # shrinking proportionally with the window the way the Visualization tab does.
        if bank_pane_present:
            splitter.setSizes([200, 300, 700])
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 0)
            splitter.setStretchFactor(2, 1)
            splitter.setCollapsible(0, False)
            splitter.setCollapsible(1, False)
            splitter.setCollapsible(2, False)
        else:
            splitter.setSizes([360, 840])
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setCollapsible(0, False)
            splitter.setCollapsible(1, False)
        layout.addWidget(splitter)
        return module_tab, out_tabs

    def update_global_dropdowns(self):
        if self.df is None: return
        cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = self.df.columns.tolist()

        def sync_list(w, items):
            sel = [i.text() for i in w.selectedItems()]
            w.clear(); w.addItems(items)
            for i in range(w.count()):
                it = w.item(i)
                is_sel = it.text() in sel
                it.setSelected(is_sel)
                # Visually empty until dragged in: only selected rows are shown.
                # The full list still exists underneath (selectedItems() is untouched),
                # so every analysis function's logic keeps working exactly as before --
                # this only changes what's visible, per the user's request to make the
                # drop targets look genuinely empty instead of pre-filled with every variable.
                it.setHidden(not is_sel)

        def sync_combo(w, items):
            cur = w.currentText()
            w.clear(); w.addItems([""] + items)
            if cur in items: w.setCurrentText(cur)

        list_widgets = []
        for list_name in ['transform_vars', 'reverse_vars', 'item_vars', 'item_cvi_vars', 'corr_vars', 'reg_block1', 'reg_block2', 'reg_block3', 'efa_vars', 'anova_dv', 'anova_covar', 'clus_vars', 'lca_vars', 'lgcm_vars', 'rm_vars', 'sna_vars']:
            if hasattr(self, list_name): list_widgets.append(getattr(self, list_name))
        for w in list_widgets: sync_list(w, cols)

        all_col_lists = []
        for list_name in ['anova_iv', 'catpca_vars', 'xtab_v1', 'xtab_v2', 'desc_list']:
            if hasattr(self, list_name): all_col_lists.append(getattr(self, list_name))
        for w in all_col_lists: sync_list(w, all_cols)
            
        combo_num = []
        for combo_name in ['t_dv', 'reg_dv', 'inter_var1', 'inter_var2', 'viz_y', 'fore_y', 'viz_m', 'med_x', 'med_m', 'med_y']:
            if hasattr(self, combo_name): combo_num.append(getattr(self, combo_name))
        for w in combo_num: sync_combo(w, cols)
            
        combo_all = []
        for combo_name in ['dummy_var', 't_iv', 'sem_var_combo', 'recode_var_combo', 'viz_x', 'fore_t', 'dup_var_combo', 'remove_filter_var']:
            if hasattr(self, combo_name): combo_all.append(getattr(self, combo_name))
        for w in combo_all: sync_combo(w, all_cols)

        # Refresh every "Available Variables" drag-source bank placed across tabs.
        if hasattr(self, 'var_banks'):
            for bank in self.var_banks:
                items = cols if getattr(bank, '_numeric_only', False) else all_cols
                bank.clear()
                bank.addItems(items)


    # ==========================================
    # MODULE 1: DATA MANAGEMENT
    # ==========================================
    def init_data_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        cl = QVBoxLayout(left_panel)
        cl.setContentsMargins(0, 0, 0, 0)

        self.add_info_box(cl, "<b>Data Management:</b> Use this section to load datasets, recode errors, reverse scores, and compute new aggregate variables (e.g., Means or Sums) before running your main analyses.")

        # --- Import / Export: kept as its own permanent row above the grouped tools,
        # since loading/saving a file is a different kind of action than transforming
        # variables and shouldn't be buried inside a sub-tab. ---
        btn_h = QHBoxLayout()
        btn_load = QPushButton("📂 Load CSV/Excel/SPSS")
        btn_load.clicked.connect(self.load_data)
        
        btn_save = QPushButton("💾 Save Dataset")
        btn_save.setStyleSheet("background-color: #10B981;")
        btn_save.clicked.connect(self.save_dataset)
        
        btn_h.addWidget(btn_load); btn_h.addWidget(btn_save)
        cl.addLayout(btn_h)

        # --- The remaining variable-transformation tools are grouped into sub-tabs
        # by intent, instead of six QGroupBoxes stacked in one long scroll. This is
        # the same QTabWidget pattern already used for Dataset/Variable view on the
        # right side of this tab, kept uniform across the app. ---
        dm_tabs = QTabWidget()

        # Tab 1: Compute & Transform -- "build a new variable from existing ones"
        compute_tab = QWidget()
        cgl_outer = QVBoxLayout(compute_tab)
        cgl_outer.setAlignment(Qt.AlignmentFlag.AlignTop)
        cgl_outer.setContentsMargins(10, 10, 10, 10)

        cg = QGroupBox("Compute Variables (Sum / Mean / Subtract / Z-Score)")
        cg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        cgl = QVBoxLayout(cg)
        cgl.setSpacing(5)
        
        self.transform_vars = QListWidget()
        self.transform_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.transform_vars.setMinimumHeight(80)
        self.setup_list_selection(cgl, "Select Variables:", self.transform_vars, "Compute Variables")
        
        self.new_var_name = QLineEdit(); self.new_var_name.setPlaceholderText("New Variable Name")
        cgl.addWidget(self.new_var_name)

        btn_grid = QGridLayout()
        btn_sum = QPushButton("Sum (Total)"); btn_sum.clicked.connect(lambda: self.compute_score('sum'))
        btn_mean = QPushButton("Mean"); btn_mean.clicked.connect(lambda: self.compute_score('mean'))
        btn_sub = QPushButton("Subtract (V1 - V2)"); btn_sub.clicked.connect(lambda: self.compute_score('subtract'))
        btn_z = QPushButton("Z-Score"); btn_z.clicked.connect(lambda: self.compute_score('z'))
        
        btn_grid.addWidget(btn_sum, 0, 0); btn_grid.addWidget(btn_mean, 0, 1)
        btn_grid.addWidget(btn_sub, 1, 0); btn_grid.addWidget(btn_z, 1, 1)
        cgl.addLayout(btn_grid)
        cgl_outer.addWidget(cg)

        dup_g = QGroupBox("Duplicate & Convert Variable")
        dup_g.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        dup_l = QFormLayout(dup_g)
        dup_l.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        dup_l.setVerticalSpacing(5)
        self.dup_var_combo = QComboBox()
        self.dup_new_name = QLineEdit(); self.dup_new_name.setPlaceholderText("New numeric variable name")
        btn_dup = QPushButton("Duplicate (Force to Numeric)")
        btn_dup.clicked.connect(self.duplicate_variable)
        dup_l.addRow("Original Variable:", self.dup_var_combo)
        dup_l.addRow("New Name:", self.dup_new_name)
        dup_l.addRow(btn_dup)
        cgl_outer.addWidget(dup_g)

        ig = QGroupBox("Create Interaction & Dummies")
        ig.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        igl = QVBoxLayout(ig)
        igl.setSpacing(5)
        
        lbl_inter = QLabel("<b>Interaction (Var1 × Var2)</b>")
        lbl_inter.setWordWrap(True); lbl_inter.setMinimumWidth(10)
        igl.addWidget(lbl_inter)
        
        ih = QHBoxLayout()
        self.inter_var1 = QComboBox(); self.inter_var2 = QComboBox()
        ih.addWidget(self.inter_var1); ih.addWidget(QLabel("×")); ih.addWidget(self.inter_var2)
        igl.addLayout(ih)
        btn_inter = QPushButton("Multiply & Create"); btn_inter.clicked.connect(self.create_interaction)
        igl.addWidget(btn_inter)
        
        lbl_dum = QLabel("<b>Create Dummy Variable</b>")
        lbl_dum.setWordWrap(True); lbl_dum.setMinimumWidth(10)
        igl.addWidget(lbl_dum)
        
        self.dummy_var = QComboBox()
        self.dummy_method = QComboBox(); self.dummy_method.addItems(["Median Split (0/1)", "Mean Split (0/1)", "One-Hot Encoding"])
        igl.addWidget(self.dummy_var); igl.addWidget(self.dummy_method)
        btn_dum = QPushButton("Generate Dummy"); btn_dum.clicked.connect(self.create_dummy)
        igl.addWidget(btn_dum)
        cgl_outer.addWidget(ig)
        cgl_outer.addStretch()

        compute_scroll = QScrollArea()
        compute_scroll.setWidgetResizable(True)
        compute_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        compute_scroll.setFrameShape(QFrame.Shape.NoFrame)
        compute_scroll.setWidget(compute_tab)
        dm_tabs.addTab(compute_scroll, "🧮 Compute && Transform")

        # Tab 2: Recode & Reverse -- "modify the values of an existing variable"
        recode_tab = QWidget()
        rcl_outer = QVBoxLayout(recode_tab)
        rcl_outer.setAlignment(Qt.AlignmentFlag.AlignTop)
        rcl_outer.setContentsMargins(10, 10, 10, 10)

        rcg = QGroupBox("Recode Variable")
        rcg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        rcl = QFormLayout(rcg)
        rcl.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        rcl.setVerticalSpacing(5)
        self.recode_var_combo = QComboBox()
        self.recode_rules = QLineEdit(); self.recode_rules.setPlaceholderText("e.g., Male=1, Female=2, 99=NaN")
        
        self.recode_new_var_chk = QCheckBox("Create as new variable")
        self.recode_new_name = QLineEdit(); self.recode_new_name.setPlaceholderText("New var name")
        self.recode_new_name.setEnabled(False)
        self.recode_new_var_chk.toggled.connect(self.recode_new_name.setEnabled)
        
        self.recode_force_num = QCheckBox("Force to Numeric (unmatched text becomes missing)")
        self.recode_force_num.setChecked(True)
        
        rcl.addRow("Variable:", self.recode_var_combo)
        rcl.addRow("Rules:", self.recode_rules)
        rcl.addRow("", self.recode_new_var_chk)
        rcl.addRow("New Name:", self.recode_new_name)
        rcl.addRow("", self.recode_force_num)
        
        btn_recode = QPushButton("Recode Values")
        btn_recode.clicked.connect(self.run_recode)
        rcl.addRow(btn_recode)
        rcl_outer.addWidget(rcg)

        rg = QGroupBox("Reverse Variables")
        rg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        rgl = QVBoxLayout(rg)
        rgl.setSpacing(5)
        
        self.reverse_vars = QListWidget()
        self.reverse_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.reverse_vars.setMinimumHeight(80)
        self.setup_list_selection(rgl, "Select Variables to Reverse:", self.reverse_vars, "Reverse Variables")
        
        rev_h = QHBoxLayout()
        self.scale_min = QSpinBox(); self.scale_min.setValue(1)
        self.scale_max = QSpinBox(); self.scale_max.setValue(5)
        rev_h.addWidget(QLabel("Min:")); rev_h.addWidget(self.scale_min)
        rev_h.addWidget(QLabel("Max:")); rev_h.addWidget(self.scale_max)
        rgl.addLayout(rev_h)
        
        self.rev_new_var_chk = QCheckBox("Create as new variable (append '_Rev')")
        self.rev_new_var_chk.setChecked(True)
        rgl.addWidget(self.rev_new_var_chk)
        
        btn_rev = QPushButton("Reverse Selected"); btn_rev.clicked.connect(self.reverse_score)
        rgl.addWidget(btn_rev)
        rcl_outer.addWidget(rg)
        rcl_outer.addStretch()

        recode_scroll = QScrollArea()
        recode_scroll.setWidgetResizable(True)
        recode_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        recode_scroll.setFrameShape(QFrame.Shape.NoFrame)
        recode_scroll.setWidget(recode_tab)
        dm_tabs.addTab(recode_scroll, "🔁 Recode && Reverse")

        # Tab 3: Remove Cases
        rc_tab = QWidget()
        rcl_outer = QVBoxLayout(rc_tab)
        rcl_outer.setAlignment(Qt.AlignmentFlag.AlignTop)
        rcl_outer.setContentsMargins(10, 10, 10, 10)
        rm_g = QGroupBox("Remove Cases (Rows)")
        rm_g.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        rm_l = QVBoxLayout(rm_g)
        rm_l.setSpacing(6)
        self.add_info_box(rm_l, "Remove rows based on a condition. Use Undo (Ctrl+Z) to reverse.")
        filter_form = QFormLayout()
        filter_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.remove_filter_var = QComboBox()
        self.remove_filter_op = QComboBox()
        self.remove_filter_op.addItems(["==", "!=", ">", ">=", "<", "<=", "contains", "is missing"])
        self.remove_filter_val = QLineEdit()
        self.remove_filter_val.setPlaceholderText("Value (leave blank for 'is missing')")
        filter_form.addRow("Variable:", self.remove_filter_var)
        filter_form.addRow("Condition:", self.remove_filter_op)
        filter_form.addRow("Value:", self.remove_filter_val)
        rm_l.addLayout(filter_form)
        btn_rm_filter = QPushButton("🗑 Remove Matching Rows")
        btn_rm_filter.clicked.connect(self.remove_cases_by_filter)
        rm_l.addWidget(btn_rm_filter)
        rm_l.addWidget(QLabel("— or remove by row number —"))
        idx_form = QFormLayout()
        idx_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.remove_row_indices = QLineEdit()
        self.remove_row_indices.setPlaceholderText("e.g. 3, 7, 15-20  (1-based row numbers)")
        idx_form.addRow("Row indices:", self.remove_row_indices)
        rm_l.addLayout(idx_form)
        btn_rm_idx = QPushButton("🗑 Remove by Row Number")
        btn_rm_idx.clicked.connect(self.remove_cases_by_index)
        rm_l.addWidget(btn_rm_idx)
        rcl_outer.addWidget(rm_g); rcl_outer.addStretch()
        rm_scroll = QScrollArea()
        rm_scroll.setWidgetResizable(True)
        rm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rm_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rm_scroll.setWidget(rc_tab)
        dm_tabs.addTab(rm_scroll, "🗑 Remove Cases")

        cl.addWidget(dm_tabs)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left_panel)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        splitter.addWidget(scroll)

        right_panel = QWidget()
        rl = QVBoxLayout(right_panel)
        rl.setContentsMargins(10, 10, 10, 10)
        
        top = QHBoxLayout()
        top.addWidget(QLabel("<h2 style='color:#4F46E5; margin:0;'>Dataset View</h2>"))
        top.addStretch()
        
        btn_add_row = QPushButton("✚ Add Row")
        btn_add_row.setObjectName("AdvBtn")
        btn_add_row.clicked.connect(self.add_manual_row)
        
        btn_add_col = QPushButton("✚ Add Variable")
        btn_add_col.setObjectName("AdvBtn")
        btn_add_col.clicked.connect(self.add_manual_variable)
        
        top.addWidget(btn_add_row)
        top.addWidget(btn_add_col)
        rl.addLayout(top)

        self.data_tabs = QTabWidget()
        self.data_table = QTableWidget()
        self.data_table.cellChanged.connect(self.update_dataframe_cell)
        self.data_tabs.addTab(self.data_table, "Dataset")

        self.var_table = QTableWidget()
        self.var_table.cellChanged.connect(self.rename_variable)
        self.data_tabs.addTab(self.var_table, "Variable")
        
        rl.addWidget(self.data_tabs)
        splitter.addWidget(right_panel)
        splitter.setSizes([380, 820])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        layout.addWidget(splitter)
        self.tabs.addWidget(tab)

    def _ensure_df_exists(self):
        if self.df is None:
            self.df = pd.DataFrame()
            self.var_labels = {}
            self.var_value_labels = {}
            self.var_scales = {}

    def add_manual_variable(self):
        self._ensure_df_exists()
        name, ok = QInputDialog.getText(self, "New Variable", "Enter variable name:")
        if not ok or not name.strip(): return
        name = name.strip()
        
        if name in self.df.columns:
            return QMessageBox.warning(self, "Warning", "Variable already exists.")
            
        self.save_state()
        if len(self.df) == 0:
            self.df[name] = pd.Series(dtype='object')
        else:
            self.df[name] = np.nan
            
        self.var_labels[name] = ""
        self.var_value_labels[name] = ""
        self.var_scales[name] = "Nominal"
        
        self.update_global_dropdowns()
        self.populate_data_tables()

    def add_manual_row(self):
        self._ensure_df_exists()
        if len(self.df.columns) == 0:
            return QMessageBox.warning(self, "Warning", "Please add a variable (column) first.")
            
        self.save_state()
        new_row = pd.DataFrame({col: [np.nan] for col in self.df.columns})
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.populate_data_tables()

    def update_dataframe_cell(self, row, col):
        if self.df is None: return
        try:
            val = self.data_table.item(row, col).text().strip()
            
            if val.lower() in ['', 'nan', 'na', 'null']:
                self.df.iat[row, col] = np.nan
            else:
                try:
                    self.df.iat[row, col] = float(val) if '.' in val else int(val)
                except ValueError:
                    self.df.iat[row, col] = val
        except Exception:
            pass

    def read_csv_robust(self, file_name):
        """
        Reads a CSV with automatic encoding and delimiter fallback. Plain
        pd.read_csv() throws an opaque UnicodeDecodeError on the very common case
        of Excel-exported CSVs (cp1252/Latin-1), and silently mis-parses
        semicolon-delimited CSVs (common in European locales) into a single
        garbage column if the delimiter isn't specified. This tries the realistic
        encodings in order of likelihood, uses csv.Sniffer to detect the actual
        delimiter (so it can be reported back accurately), and returns
        (dataframe, encoding_used, delimiter_used). Raises the last exception if
        nothing works, so the caller's except block still surfaces a clear message.
        """
        import csv as _csv
        # A UTF-8 BOM is the one case plain 'utf-8' will *decode without error* but
        # get subtly wrong (the BOM marker leaks into the first column's name instead
        # of being stripped) -- so detect it up front and prioritize utf-8-sig.
        try:
            with open(file_name, 'rb') as fh:
                raw_start = fh.read(4)
        except Exception:
            raw_start = b''
        has_bom = raw_start.startswith(b'\xef\xbb\xbf')
        encodings_to_try = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1'] if has_bom else ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']

        last_err = None
        for enc in encodings_to_try:
            try:
                with open(file_name, 'r', encoding=enc, newline='') as fh:
                    sample = fh.read(8192)
                try:
                    delim = _csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
                except Exception:
                    delim = ','
                df = pd.read_csv(file_name, encoding=enc, sep=delim)
                if df.shape[1] == 1 and df.shape[0] > 0:
                    # Sniffer guessed wrong; retry forcing the other common delimiters.
                    for forced_sep in [';', ',', '\t', '|']:
                        if forced_sep == delim:
                            continue
                        try:
                            df_try = pd.read_csv(file_name, encoding=enc, sep=forced_sep)
                            if df_try.shape[1] > 1:
                                return df_try, enc, forced_sep
                        except Exception:
                            continue
                return df, enc, delim
            except (UnicodeDecodeError, UnicodeError) as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        raise last_err if last_err is not None else IOError("Could not read CSV file with any supported encoding.")

    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "Data Files (*.csv *.xlsx *.xls *.sav)")
        if file_name:
            try:
                self.save_state()
                encoding_note = ""
                if file_name.endswith('.csv'):
                    self.df, enc_used, delim_used = self.read_csv_robust(file_name)
                    if enc_used != 'utf-8' or delim_used != ',':
                        encoding_note = f"\n(Detected encoding: {enc_used}, delimiter: '{delim_used}')"
                elif file_name.endswith(('.xlsx', '.xls')): 
                    xl = pd.ExcelFile(file_name)
                    if len(xl.sheet_names) > 1:
                        sheet, ok = QInputDialog.getItem(self, "Select Sheet", "Multiple sheets found. Select one to load:", xl.sheet_names, 0, False)
                        if ok and sheet:
                            self.df = pd.read_excel(file_name, sheet_name=sheet)
                        else:
                            self.undo_data()
                            return 
                    else:
                        self.df = pd.read_excel(file_name)
                elif file_name.endswith('.sav'): 
                    try:
                        self.df = pd.read_spss(file_name)
                    except Exception:
                        return QMessageBox.critical(self, "Missing Module", "To read SPSS files, please run: pip install pyreadstat")
                
                # Initialize label storage for new data
                self.var_labels = {col: "" for col in self.df.columns}
                self.var_value_labels = {col: "" for col in self.df.columns}
                self.var_scales = {}
                
                for col in self.df.columns:
                    if self.df[col].dtype == 'object': self.var_scales[col] = "Nominal"
                    elif self.df[col].dtype in ['float64', 'float32']: self.var_scales[col] = "Ratio"
                    else: self.var_scales[col] = "Interval"

                self.update_global_dropdowns()
                self.populate_data_tables()
                QMessageBox.information(self, "Success", f"Data loaded successfully.\nRows: {len(self.df)}\nColumns: {len(self.df.columns)}{encoding_note}")
            except UnicodeDecodeError as e:
                QMessageBox.critical(self, "Encoding Error", f"Could not read this file with any common encoding (UTF-8, CP1252, Latin-1).\nThe file may be corrupted or use an unsupported encoding.\n\nDetails: {str(e)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load data:\n{str(e)}")

    def save_dataset(self):
        if self.df is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Dataset", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if path:
            try:
                if path.endswith('.csv'): self.df.to_csv(path, index=False)
                elif path.endswith('.xlsx'): self.df.to_excel(path, index=False)
                QMessageBox.information(self, "Success", "Dataset successfully saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file:\n{str(e)}")

    def populate_data_tables(self):
        if self.df is None: return
        
        limit = min(10000, len(self.df))
        self.data_table.blockSignals(True)
        self.data_table.clear()
        self.data_table.setRowCount(limit)
        self.data_table.setColumnCount(len(self.df.columns))
        self.data_table.setHorizontalHeaderLabels(self.df.columns.astype(str))
        for i in range(limit):
            for j in range(len(self.df.columns)):
                self.data_table.setItem(i, j, QTableWidgetItem(str(self.df.iat[i, j])))
        self.data_table.blockSignals(False)
                
        self.var_table.blockSignals(True)
        self.var_table.clearContents()
        self.var_table.setColumnCount(6)
        self.var_table.setHorizontalHeaderLabels(["Variable Name", "Variable Label", "Value Labels (e.g., 1=Low)", "Measurement Scale", "Data Type", "Missing Values"])
        
        self.var_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.var_table.horizontalHeader().setStretchLastSection(True)
        
        self.var_table.setRowCount(len(self.df.columns))
        
        for i, col in enumerate(self.df.columns):
            self.var_table.setItem(i, 0, QTableWidgetItem(str(col)))
            self.var_table.setItem(i, 1, QTableWidgetItem(self.var_labels.get(col, "")))
            self.var_table.setItem(i, 2, QTableWidgetItem(self.var_value_labels.get(col, "")))
            
            scale_combo = QComboBox()
            scale_combo.addItems(["Nominal", "Ordinal", "Interval", "Ratio"])
            scale_combo.setCurrentText(self.var_scales.get(col, "Interval"))
            scale_combo.currentTextChanged.connect(lambda text, c=col: self.update_scale(c, text))
            self.var_table.setCellWidget(i, 3, scale_combo)
            
            # Interactive Type Casting Box
            type_combo = QComboBox()
            type_combo.addItems(["Numeric (Float)", "Integer", "String/Text", "Category"])
            current_dtype = str(self.df[col].dtype)
            if 'float' in current_dtype: type_combo.setCurrentText("Numeric (Float)")
            elif 'int' in current_dtype: type_combo.setCurrentText("Integer")
            elif 'category' in current_dtype: type_combo.setCurrentText("Category")
            else: type_combo.setCurrentText("String/Text")
            type_combo.currentTextChanged.connect(lambda text, c=col: self.cast_data_type(c, text))
            self.var_table.setCellWidget(i, 4, type_combo)
            
            m_item = QTableWidgetItem(str(self.df[col].isna().sum()))
            m_item.setFlags(m_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.var_table.setItem(i, 5, m_item)
            
        self.var_table.setColumnWidth(0, 150)
        self.var_table.setColumnWidth(1, 200)
        self.var_table.setColumnWidth(2, 250) 
        self.var_table.setColumnWidth(4, 150) 
        
        self.var_table.blockSignals(False)

    def update_scale(self, col, text):
        self.var_scales[col] = text

    def cast_data_type(self, col, type_str):
        try:
            self.save_state()
            if type_str == "Numeric (Float)":
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(float)
            elif type_str == "Integer":
                # 'Int64' allows for missing NA values in integer columns
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')
            elif type_str == "String/Text":
                self.df[col] = self.df[col].astype(str)
            elif type_str == "Category":
                self.df[col] = self.df[col].astype('category')
                
            self.update_global_dropdowns()
            self.populate_data_tables()
            QMessageBox.information(self, "Success", f"Variable '{col}' has been converted to {type_str}.")
        except Exception as e:
            self.undo_data()
            QMessageBox.critical(self, "Error", f"Could not change type for {col}:\n{str(e)}")

    def rename_variable(self, row, col):
        if self.df is None: return
        old_name = self.df.columns[row]
        
        if col == 0: 
            new_name = self.var_table.item(row, col).text().strip()
            if new_name and old_name != new_name and new_name not in self.df.columns:
                self.save_state()
                self.df.rename(columns={old_name: new_name}, inplace=True)
                self.var_labels[new_name] = self.var_labels.pop(old_name, "")
                self.var_value_labels[new_name] = self.var_value_labels.pop(old_name, "")
                self.var_scales[new_name] = self.var_scales.pop(old_name, "Interval")
                self.update_global_dropdowns()
                self.populate_data_tables()
        elif col == 1: 
            new_label = self.var_table.item(row, col).text().strip()
            self.var_labels[old_name] = new_label
        elif col == 2:
            new_val_label = self.var_table.item(row, col).text().strip()
            self.var_value_labels[old_name] = new_val_label

    def duplicate_variable(self):
        if self.df is None: return
        orig_col = self.dup_var_combo.currentText()
        new_col = self.dup_new_name.text().strip()
        
        if not orig_col or not new_col: 
            return QMessageBox.warning(self, "Warning", "Please select a variable and provide a new name.")
            
        if new_col in self.df.columns:
            return QMessageBox.warning(self, "Warning", "A variable with this new name already exists.")
            
        try:
            self.save_state()
            # Force conversion to numeric (non-string). Invalid text will be cleanly converted to NaN.
            self.df[new_col] = pd.to_numeric(self.df[orig_col], errors='coerce')
            
            # Carry over labels
            self.var_labels[new_col] = self.var_labels.get(orig_col, "") + " (Copy)"
            self.var_value_labels[new_col] = self.var_value_labels.get(orig_col, "")
            self.var_scales[new_col] = "Interval" if self.df[new_col].nunique() > 5 else "Nominal"
            
            self.update_global_dropdowns()
            self.populate_data_tables()
            QMessageBox.information(self, "Success", f"Variable '{orig_col}' was duplicated as '{new_col}' and successfully forced to a numeric type.")
        except Exception as e:
            self.undo_data()
            QMessageBox.critical(self, "Error", f"Could not duplicate variable: {str(e)}")
            
    def run_recode(self):
        var = self.recode_var_combo.currentText()
        rules_raw = self.recode_rules.text().strip()
        
        if not var or not rules_raw: return
        
        try:
            self.save_state()
            target_col = var
            if self.recode_new_var_chk.isChecked():
                target_col = self.recode_new_name.text().strip()
                if not target_col: 
                    self.undo_data()
                    return QMessageBox.warning(self, "Warning", "Please provide a name for the new recoded variable.")

            # Parse bulk mapping rules
            rules = [r.strip() for r in rules_raw.split(',') if r.strip()]
            mapping = {}
            for r in rules:
                if '=' not in r:
                    self.undo_data()
                    return QMessageBox.warning(self, "Warning", f"Invalid rule format: '{r}'. Please use Old=New format (e.g., Male=1, Female=2).")
                old_v, new_v = r.split('=', 1)
                mapping[old_v.strip().strip("'\"")] = new_v.strip().strip("'\"")

            # Safely create the target column if it doesn't exist
            if target_col not in self.df.columns:
                self.df[target_col] = self.df[var].copy()
                
            # Drop categorical restrictions immediately by converting to object
            self.df[target_col] = self.df[target_col].astype(object)
            
            replaced_total = 0
            
            for old_clean, new_clean in mapping.items():
                is_nan_old = old_clean.lower() in ['nan', 'none', 'null', 'na', '']
                is_nan_new = new_clean.lower() in ['nan', 'none', 'null', 'na', '']

                if is_nan_new:
                    new_val = np.nan
                else:
                    try:
                        new_val = float(new_clean) if '.' in new_clean else int(new_clean)
                    except ValueError:
                        new_val = new_clean

                # Vectorized Mask Matching (Highly Robust)
                if is_nan_old:
                    mask = self.df[var].isna()
                else:
                    # Match strings robustly (ignores leading/trailing spaces and case)
                    mask = self.df[var].astype(str).str.strip().str.lower() == old_clean.lower()
                    
                    # Add numeric equivalency if applicable
                    try:
                        num_old = float(old_clean)
                        mask_num = pd.to_numeric(self.df[var], errors='coerce') == num_old
                        mask = mask | mask_num
                    except ValueError:
                        pass

                replaced_count = mask.sum()
                replaced_total += replaced_count
                
                # Apply the new value directly via the mask
                self.df.loc[mask, target_col] = new_val

            if replaced_total == 0:
                self.undo_data()
                return QMessageBox.warning(self, "No Matches Found", f"Could not find any values matching your rules in '{var}'. Check for typos.")

            # Core Fix: Explicit user control to force numeric conversion
            if self.recode_force_num.isChecked():
                self.df[target_col] = pd.to_numeric(self.df[target_col], errors='coerce')
            else:
                self.df[target_col] = pd.to_numeric(self.df[target_col], errors='ignore')

            if target_col not in self.var_labels:
                self.var_labels[target_col] = self.var_labels.get(var, "")
                self.var_value_labels[target_col] = self.var_value_labels.get(var, "")

            self.update_global_dropdowns()
            self.populate_data_tables()
            QMessageBox.information(self, "Success", f"Successfully recoded {replaced_total} instance(s) in {target_col}.")
        except Exception as e:
            self.undo_data()
            QMessageBox.critical(self, "Error", f"Recoding Failed: {str(e)}")

    def compute_score(self, method):
        if self.df is None: return
        selected = [item.text() for item in self.transform_vars.selectedItems()]
        if not selected: return
        self.save_state()
        try:
            if method == 'z':
                for col in selected: 
                    self.df[f"{col}_Z"] = stats.zscore(self.df[col].dropna())
                    self.var_scales[f"{col}_Z"] = "Ratio"
            elif method == 'subtract':
                if len(selected) != 2:
                    self.undo_data() 
                    return QMessageBox.warning(self, "Warning", "Please select exactly TWO variables to perform a subtraction (Var1 - Var2).")
                
                new_name = self.new_var_name.text().strip()
                if not new_name: 
                    self.undo_data()
                    return QMessageBox.warning(self, "Warning", "Please provide a New Variable Name.")
                
                self.df[new_name] = self.df[selected[0]] - self.df[selected[1]]
                self.var_scales[new_name] = "Interval"
            else:
                new_name = self.new_var_name.text().strip()
                if not new_name: 
                    self.undo_data()
                    return QMessageBox.warning(self, "Warning", "Please provide a New Variable Name.")
                
                if method == 'sum': self.df[new_name] = self.df[selected].sum(axis=1)
                elif method == 'mean': self.df[new_name] = self.df[selected].mean(axis=1)
                self.var_scales[new_name] = "Interval"
                
            self.update_global_dropdowns()
            self.populate_data_tables()
        except Exception as e: 
            QMessageBox.critical(self, "Error", str(e))

    def create_interaction(self):
        if self.df is None: return
        v1 = self.inter_var1.currentText()
        v2 = self.inter_var2.currentText()
        if not v1 or not v2: return
        self.save_state()
        try:
            new_name = f"{v1}_x_{v2}"
            self.df[new_name] = self.df[v1] * self.df[v2]
            self.var_scales[new_name] = "Ratio"
            self.update_global_dropdowns()
            self.populate_data_tables()
            QMessageBox.information(self, "Success", f"Interaction variable created: {new_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def reverse_score(self):
        if self.df is None: return
        selected = [item.text() for item in self.reverse_vars.selectedItems()]
        if not selected: return
        self.save_state()
        for col in selected: 
            new_col = f"{col}_Rev" if self.rev_new_var_chk.isChecked() else col
            self.df[new_col] = (self.scale_max.value() + self.scale_min.value()) - self.df[col]
            self.var_scales[new_col] = self.var_scales.get(col, "Interval")
        self.update_global_dropdowns()
        self.populate_data_tables()
        QMessageBox.information(self, "Success", "Reversed scoring applied successfully.")

    def remove_cases_by_filter(self):
        if self.df is None: return
        col = self.remove_filter_var.currentText()
        op = self.remove_filter_op.currentText()
        val_text = self.remove_filter_val.text().strip()
        if not col: return
        try:
            self.save_state()
            n_before = len(self.df)
            if op == "is missing":
                mask = self.df[col].isna()
            elif op == "contains":
                mask = self.df[col].astype(str).str.contains(val_text, na=False)
            else:
                try:
                    num_val = float(val_text)
                    col_s = pd.to_numeric(self.df[col], errors='coerce')
                    ops = {"==": col_s == num_val, "!=": col_s != num_val, ">": col_s > num_val,
                           ">=": col_s >= num_val, "<": col_s < num_val, "<=": col_s <= num_val}
                    mask = ops.get(op, pd.Series([False]*len(self.df)))
                except ValueError:
                    if op == "==": mask = self.df[col].astype(str) == val_text
                    elif op == "!=": mask = self.df[col].astype(str) != val_text
                    else:
                        QMessageBox.warning(self, "Error", f"Operator '{op}' requires a numeric value.")
                        self.undo_data(); return
            self.df = self.df[~mask].reset_index(drop=True)
            n_removed = n_before - len(self.df)
            self.update_global_dropdowns(); self.populate_data_tables()
            QMessageBox.information(self, "Done", f"Removed {n_removed} case(s). Dataset now has {len(self.df)} rows.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e)); self.undo_data()

    def remove_cases_by_index(self):
        if self.df is None: return
        idx_text = self.remove_row_indices.text().strip()
        if not idx_text: return
        try:
            self.save_state()
            rows_to_drop = set()
            for part in idx_text.replace(' ', '').split(','):
                if '-' in part:
                    lo, hi = part.split('-', 1)
                    rows_to_drop.update(range(int(lo) - 1, int(hi)))
                else:
                    rows_to_drop.add(int(part) - 1)
            valid_rows = [r for r in rows_to_drop if 0 <= r < len(self.df)]
            n_before = len(self.df)
            self.df = self.df.drop(index=valid_rows).reset_index(drop=True)
            n_removed = n_before - len(self.df)
            self.update_global_dropdowns(); self.populate_data_tables()
            QMessageBox.information(self, "Done", f"Removed {n_removed} row(s). Dataset now has {len(self.df)} rows.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e)); self.undo_data()


    def create_dummy(self):
        if self.df is None: return
        col = self.dummy_var.currentText()
        method = self.dummy_method.currentText()
        if not col: return
        self.save_state()
        try:
            if "One-Hot" in method:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=False).astype(int)
                self.df = pd.concat([self.df, dummies], axis=1)
                for c in dummies.columns: self.var_scales[c] = "Nominal"
            elif "Median" in method:
                new_col = f"{col}_MedSplit"
                self.df[new_col] = (self.df[col] > self.df[col].median()).astype(int)
                self.var_scales[new_col] = "Nominal"
            elif "Mean" in method:
                new_col = f"{col}_MeanSplit"
                self.df[new_col] = (self.df[col] > self.df[col].mean()).astype(int)
                self.var_scales[new_col] = "Nominal"
                
            self.update_global_dropdowns()
            self.populate_data_tables()
            QMessageBox.information(self, "Success", f"Dummy variables created for {col}.")
        except Exception as e: 
            QMessageBox.critical(self, "Error", str(e))


    # ==========================================
    # MODULE 1.5: DATA VISUALIZATION
    # ==========================================
    def init_viz_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.add_info_box(cl, "<b>Data Visualization (Graph Builder):</b> Create robust diagnostic and interaction charts. Select your plot type and variables, and customize the aesthetics to match your preferences.")


        dg = QGroupBox("Data Selection")
        dg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        form = QFormLayout(dg)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        form.setVerticalSpacing(5)
        
        self.viz_type = QComboBox()
        self.viz_type.addItems(["Histogram", "Scatterplot", "Bar Chart", "Line Plot", "Moderation Chart (Interaction)"])
        self.viz_x = QComboBox()
        self.viz_y = QComboBox()
        self.viz_m = QComboBox()
        self.viz_m.setEnabled(False)
        self.enable_drag_drop_combo(self.viz_x)
        self.enable_drag_drop_combo(self.viz_y)
        self.enable_drag_drop_combo(self.viz_m)
        
        self.viz_type.currentTextChanged.connect(lambda t: self.viz_m.setEnabled(t == "Moderation Chart (Interaction)"))
        
        form.addRow("Plot Type:", self.viz_type)
        form.addRow("X-Axis Variable:", self.viz_x)
        form.addRow("Y-Axis (Target Variable):", self.viz_y)
        form.addRow("Moderator (Z) Variable:", self.viz_m)
        cl.addWidget(dg)
        
        ag = QGroupBox("Aesthetics & Design")
        ag.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        aform = QFormLayout(ag)
        aform.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        aform.setVerticalSpacing(5)
        self.viz_palette = QComboBox()
        self.viz_palette.addItems(["Standard", "Pastel", "Seaborn", "Monochrome"])
        self.viz_grid = QCheckBox("Show Gridlines"); self.viz_grid.setChecked(True)
        self.viz_alpha = QSlider(Qt.Orientation.Horizontal)
        self.viz_alpha.setRange(20, 100); self.viz_alpha.setValue(70)
        
        aform.addRow("Color Palette:", self.viz_palette)
        aform.addRow("Gridlines:", self.viz_grid)
        aform.addRow("Transparency:", self.viz_alpha)
        cl.addWidget(ag)
        
        btn_row = QHBoxLayout()
        btn = QPushButton("▶ Generate Graph")
        btn.setStyleSheet("margin-top: 10px;")
        btn.clicked.connect(self.run_viz)
        btn_row.addWidget(btn)

        btn_zoom = QPushButton("↗ Pop Out Chart")
        btn_zoom.setStyleSheet("margin-top: 10px; background-color: #374151;")
        btn_zoom.setToolTip("Opens the current chart in a new, larger, resizable window with full zoom/pan/save controls.")
        btn_zoom.clicked.connect(self.pop_out_viz)
        btn_row.addWidget(btn_zoom)

        cl.addLayout(btn_row)
        cl.addStretch()
        
        tab, self.viz_tabs = self.create_split_module(cw, bank=self.build_bank_panel())
        self.tabs.addWidget(tab)

    def pop_out_viz(self):
        """
        Opens the currently displayed chart in a standalone, resizable window with
        matplotlib's full navigation toolbar (pan, zoom-rectangle, home, save).
        Works by finding the FigureCanvasQTAgg inside the current result tab's
        zoomable-chart wrapper, then creating a fresh QDialog that displays the
        same underlying matplotlib Figure at a much larger size.
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        current_widget = self.viz_tabs.currentWidget()
        if current_widget is None:
            QMessageBox.information(self, "No Chart", "Generate a chart first, then click Pop Out.")
            return

        # The zoomable chart wrapper is: QWidget → [NavigationToolbar, FigureCanvasQTAgg]
        # Walk the child widgets to find the canvas.
        canvas_widget = None
        for child in current_widget.findChildren(FigureCanvas):
            canvas_widget = child
            break

        if canvas_widget is None:
            QMessageBox.information(self, "No Chart", "The current tab doesn't contain a zoomable matplotlib chart.")
            return

        fig = canvas_widget.figure

        dlg = QDialog(self)
        dlg.setWindowTitle("Chart — Zoomed View")
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        dlg.resize(900, 650)
        dlg.setSizeGripEnabled(True)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        # Fresh canvas for the pop-out — same figure object, new canvas instance.
        # This is safe because FigureCanvasQTAgg stores a reference to the Figure,
        # not a copy; drawing calls on this canvas render the same figure.
        new_canvas = FigureCanvas(fig)
        new_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        new_canvas.updateGeometry()

        toolbar = NavigationToolbar(new_canvas, dlg)
        toolbar.setIconSize(QSize(20, 20))

        layout.addWidget(toolbar)
        layout.addWidget(new_canvas)

        dlg.show()
        new_canvas.draw()

        # Keep the dialog alive by adding it to the global windows list used elsewhere
        GLOBAL_WINDOWS.append(dlg)
        dlg.finished.connect(lambda: GLOBAL_WINDOWS.remove(dlg) if dlg in GLOBAL_WINDOWS else None)

    def run_viz(self):
        if not MATPLOTLIB_AVAILABLE or self.df is None: return
        ptype = self.viz_type.currentText()
        x_var = self.viz_x.currentText()
        y_var = self.viz_y.currentText()
        m_var = self.viz_m.currentText()
        if not x_var: return
        
        try:
            fig = Figure(figsize=(7,5))
            ax = fig.add_subplot(111)
            
            alpha_val = self.viz_alpha.value() / 100.0
            palette_name = self.viz_palette.currentText()
            
            if palette_name == "Standard": colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
            elif palette_name == "Pastel": colors = ['#A78BFA', '#34D399', '#FBBF24', '#F87171', '#60A5FA']
            elif palette_name == "Seaborn": colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
            else: colors = ['#374151', '#6B7280', '#9CA3AF', '#D1D5DB', '#F3F4F6']

            if self.is_dark_mode:
                fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#374151')
                ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                ax.title.set_color('white'); ax.tick_params(colors='white')
                main_color = colors[0] if palette_name != "Monochrome" else '#CBD5E1'
            else:
                main_color = colors[0]

            if self.viz_grid.isChecked():
                ax.grid(True, linestyle='--', alpha=0.5, color='#9CA3AF' if not self.is_dark_mode else '#4B5563')

            if ptype == "Histogram":
                data = self.df[x_var].dropna()
                ax.hist(data, bins=20, color=main_color, edgecolor='black', alpha=alpha_val)
                ax.set_xlabel(x_var); ax.set_ylabel("Frequency")
                
            elif ptype == "Scatterplot":
                if not y_var: return
                data = self.df[[x_var, y_var]].dropna()
                ax.scatter(data[x_var], data[y_var], color=main_color, alpha=alpha_val)
                self.plot_smooth_fit(ax, data[x_var], data[y_var], '#EF4444')
                ax.set_xlabel(x_var); ax.set_ylabel(y_var)
                
            elif ptype == "Bar Chart":
                data = self.df[x_var].value_counts()
                ax.bar(data.index.astype(str), data.values, color=main_color, edgecolor='black', alpha=alpha_val)
                ax.set_xlabel(x_var); ax.set_ylabel("Count")
                
            elif ptype == "Line Plot":
                if not y_var: return
                data = self.df[[x_var, y_var]].dropna()
                ax.plot(data[x_var], data[y_var], color=main_color, marker='o', alpha=alpha_val)
                ax.set_xlabel(x_var); ax.set_ylabel(y_var)
                
            elif ptype == "Moderation Chart (Interaction)":
                if not y_var or not m_var:
                    raise ValueError("Moderation Charts require X, Y, and a Moderator (Z) variable.")
                data = self.df[[x_var, y_var, m_var]].dropna()
                
                if pd.api.types.is_numeric_dtype(data[m_var]) and data[m_var].nunique() > 5:
                    mean_val = data[m_var].mean()
                    std_val = data[m_var].std()
                    data['Mod_Cat'] = np.select(
                        [data[m_var] > (mean_val + std_val), data[m_var] < (mean_val - std_val)],
                        ['High (+1 SD)', 'Low (-1 SD)'], 
                        default='Average (Mean)'
                    )
                else:
                    data['Mod_Cat'] = data[m_var].astype(str)
                
                x_is_numeric = pd.api.types.is_numeric_dtype(data[x_var])
                if not x_is_numeric:
                    data['X_Numeric'] = data[x_var].astype('category').cat.codes
                    plot_x = 'X_Numeric'
                    ax.set_xticks(data['X_Numeric'].unique())
                    ax.set_xticklabels(data[x_var].unique())
                else:
                    plot_x = x_var

                for i, cat in enumerate(data['Mod_Cat'].unique()):
                    subset = data[data['Mod_Cat'] == cat]
                    c = colors[i % len(colors)]
                    ax.scatter(subset[plot_x], subset[y_var], label=f"{m_var}: {cat}", color=c, alpha=alpha_val)
                    if len(subset) > 2:
                        self.plot_smooth_fit(ax, subset[plot_x], subset[y_var], c)
                        
                ax.set_xlabel(x_var); ax.set_ylabel(y_var)
                ax.legend(facecolor='#374151' if self.is_dark_mode else 'white', labelcolor='white' if self.is_dark_mode else 'black')

            title_str = f"{ptype} of {x_var}"
            if y_var and ptype not in ["Histogram", "Bar Chart"]: title_str += f" and {y_var}"
            if ptype == "Moderation Chart (Interaction)": title_str += f" moderated by {m_var}"
            ax.set_title(title_str, fontweight='bold')
            
            fig.tight_layout()
            chart = self.make_zoomable_chart(fig)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.viz_tabs.addTab(chart, f"Graph ({timestamp})")
            self.viz_tabs.setCurrentIndex(self.viz_tabs.count() - 1)
        except Exception as e:
            self.viz_tabs.addTab(QTextEdit(str(e)), "Error")


    # ==========================================
    # MODULE 2: DESCRIPTIVES & CROSSTABS
    # ==========================================
    def init_descriptives_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.add_info_box(cl, "<b>Descriptives & Crosstabs:</b> Use this to get an overview of your data (Means, Standard Deviations, Normality). Crosstabs allow you to see the breakdown of two categorical variables and test if they are dependent using Chi-Square.")


        dg = QGroupBox("Descriptive Statistics")
        dg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        dgl = QVBoxLayout(dg)
        dgl.setSpacing(5)
        
        self.desc_list = QListWidget()
        self.desc_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.desc_list.setMinimumHeight(120)
        self.setup_list_selection(dgl, "Variables to Analyze:", self.desc_list, "Descriptives")
        
        self.chk_desc = QCheckBox("Generate Descriptives (Mean, SD, etc.)"); self.chk_desc.setChecked(True)
        self.chk_freq = QCheckBox("Generate Frequencies")
        dgl.addWidget(self.chk_desc); dgl.addWidget(self.chk_freq)
        
        btn = QPushButton("▶ Run Selected Analysis")
        btn.clicked.connect(self.run_descriptives)
        dgl.addWidget(btn)
        cl.addWidget(dg)
        
        cg = QGroupBox("Crosstabs (Contingency Table)")
        cg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        cgl = QVBoxLayout(cg)
        cgl.setSpacing(5)
        
        self.xtab_v1 = QListWidget()
        self.xtab_v1.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.xtab_v1.setMinimumHeight(80)
        self.setup_list_selection(cgl, "Row Variables:", self.xtab_v1, "Row Variables")
        
        self.xtab_v2 = QListWidget()
        self.xtab_v2.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.xtab_v2.setMinimumHeight(80)
        self.setup_list_selection(cgl, "Column Variables:", self.xtab_v2, "Column Variables")
        
        opts = QHBoxLayout()
        self.chk_xtab_row = QCheckBox("Row Percentages")
        self.chk_xtab_col = QCheckBox("Column Percentages")
        self.chk_xtab_tot = QCheckBox("Total Percentages")
        self.chk_xtab_chi = QCheckBox("Chi-Square Test")
        self.chk_xtab_chi.setChecked(True)
        
        opts.addWidget(self.chk_xtab_row); opts.addWidget(self.chk_xtab_col)
        opts.addWidget(self.chk_xtab_tot); opts.addWidget(self.chk_xtab_chi)
        cgl.addLayout(opts)
        
        btn_xtab = QPushButton("▶ Run Crosstab")
        btn_xtab.clicked.connect(self.run_crosstab)
        cgl.addWidget(btn_xtab)
        cl.addWidget(cg)

        tab, self.desc_tabs = self.create_split_module(cw, bank=self.build_bank_panel())
        self.tabs.addWidget(tab)

    def run_descriptives(self):
        if self.df is None: return
        selected = [item.text() for item in self.desc_list.selectedItems()]
        if not selected: return

        numeric_cols = [c for c in selected if pd.api.types.is_numeric_dtype(self.df[c])]
        nonnumeric_cols = [c for c in selected if c not in numeric_cols]

        out = self.get_apa_css()

        if self.chk_desc.isChecked():
            if numeric_cols:
                # Listwise across the numeric subset only, so a string variable's
                # missingness never distorts the descriptive stats of an unrelated
                # numeric variable (and vice versa).
                data_num = self.df[numeric_cols].dropna()
                out += self.build_missing_data_note(self.df, numeric_cols, len(data_num))
                out += "<h2>Descriptive Statistics</h2><table class='apa'><tr><th>Variable</th><th>N</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th><th>Skewness</th><th>Kurtosis</th></tr>"
                desc_apa_payloads = []
                for col in numeric_cols:
                    d = data_num[col]
                    sk, ku = stats.skew(d, bias=False), stats.kurtosis(d, bias=False)
                    sk_c = "inherit" if abs(sk) <= 2 else "#EF4444"
                    ku_c = "inherit" if abs(ku) <= 7 else "#EF4444"
                    out += f"<tr><td style='text-align:left;'><b>{col}</b></td><td>{len(d)}</td><td>{self.fmt(d.mean())}</td><td>{self.fmt(d.std())}</td><td>{self.fmt(d.min())}</td><td>{self.fmt(d.max())}</td><td style='color:{sk_c}'>{self.fmt(sk)}</td><td style='color:{ku_c}'>{self.fmt(ku)}</td></tr>"

                    shapiro_p = None
                    if 3 <= len(d) <= 5000:
                        try:
                            _, shapiro_p = stats.shapiro(d)
                        except Exception:
                            shapiro_p = None
                    desc_apa_payloads.append({"var": col, "n": len(d), "mean": d.mean(), "sd": d.std(), "shapiro_p": shapiro_p})
                out += "</table><div class='interpret'><i>Note.</i> <b>Normality Guidelines (Kim, 2013):</b><br>For psychometric evaluation, an absolute Skewness value &lt; 2 and Kurtosis &lt; 7 suggest that departure from normality is not severe enough to distort parametric tests.</div>"

                for payload in desc_apa_payloads:
                    out += self.build_apa_writeup("Descriptives", payload)

            if nonnumeric_cols:
                out += (
                    f"<p style='color:#6B7280; font-size:13px;'><i>Mean/SD/Skewness/Kurtosis were not computed for "
                    f"{', '.join(nonnumeric_cols)} (not numeric). Use Frequencies below for these variables, or "
                    f"Crosstabs to cross-tabulate them against another variable.</i></p>"
                )

        if self.chk_freq.isChecked():
            out += "<h2>Frequency Tables</h2>"
            for col in selected:
                d = self.df[col].dropna()
                counts = d.value_counts().sort_index()
                percs = (counts / len(d) * 100).round(2) if len(d) > 0 else counts
                out += f"<h3>{col}</h3><table class='apa'><tr><th>Value</th><th>Count</th><th>Percent (%)</th></tr>"
                for val, cnt in counts.items():
                    out += f"<tr><td>{val}</td><td>{cnt}</td><td>{percs[val]}%</td></tr>"
                out += "</table>"
                
        if not self.chk_desc.isChecked() and not self.chk_freq.isChecked(): return
        
        tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.desc_tabs.addTab(tv, f"Descriptives ({timestamp})")
        self.desc_tabs.setCurrentIndex(self.desc_tabs.count() - 1)

    def run_crosstab(self):
        if self.df is None: return
        rows = [item.text() for item in self.xtab_v1.selectedItems()]
        cols = [item.text() for item in self.xtab_v2.selectedItems()]
        if not rows or not cols: return
        
        out = self.get_apa_css() + f"<h2>Crosstabs Analysis</h2>"
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        for v1 in rows:
            for v2 in cols:
                if v1 == v2: continue
                
                data = self.df[[v1, v2]].dropna()
                if data.empty: continue
                
                out += f"<h3>Crosstab: {v1} (Row) by {v2} (Column)</h3>"
                xtab_cnt = pd.crosstab(data[v1], data[v2], margins=True, margins_name="Total")
                
                out += "<table class='apa'><tr><th></th>"
                for col in xtab_cnt.columns: out += f"<th>{col}</th>"
                out += "</tr>"
                
                for idx in xtab_cnt.index:
                    out += f"<tr><td style='text-align:left; font-weight:bold;'>{idx}</td>"
                    for col in xtab_cnt.columns:
                        count_val = xtab_cnt.loc[idx, col]
                        cell_text = f"{count_val}"
                        
                        if self.chk_xtab_row.isChecked() and xtab_cnt.loc[idx, 'Total'] > 0:
                            pct = (count_val / xtab_cnt.loc[idx, 'Total']) * 100
                            cell_text += f"<br><span style='font-size:12px; color:#6B7280;'>{pct:.1f}% (Row)</span>"
                        if self.chk_xtab_col.isChecked() and xtab_cnt.loc['Total', col] > 0:
                            pct = (count_val / xtab_cnt.loc['Total', col]) * 100
                            cell_text += f"<br><span style='font-size:12px; color:#6B7280;'>{pct:.1f}% (Col)</span>"
                        if self.chk_xtab_tot.isChecked() and xtab_cnt.loc['Total', 'Total'] > 0:
                            pct = (count_val / xtab_cnt.loc['Total', 'Total']) * 100
                            cell_text += f"<br><span style='font-size:12px; color:#6B7280;'>{pct:.1f}% (Tot)</span>"
                            
                        out += f"<td>{cell_text}</td>"
                    out += "</tr>"
                out += "</table>"
                
                if self.chk_xtab_chi.isChecked():
                    xtab_no_margins = pd.crosstab(data[v1], data[v2])
                    try:
                        chi2, p, dof, ex = stats.chi2_contingency(xtab_no_margins)
                        out += f"<h4>Chi-Square Test of Independence</h4><table class='apa'><tr><th>Statistic</th><th>Value</th><th>df</th><th>p-value</th></tr>"
                        p_str = f"<span class='sig'>{self.fmt(p, True)}</span>" if p < 0.05 else self.fmt(p, True)
                        out += f"<tr><td>Pearson Chi-Square</td><td>{self.fmt(chi2)}</td><td>{dof}</td><td>{p_str}</td></tr></table>"
                    except:
                        out += "<p class='warn'>Chi-Square test could not be computed for this combination.</p>"
                out += "<hr>"
                
        out += "<div class='interpret'><i>Note.</i> <b>Interpretation:</b> A significant Pearson Chi-Square (p &lt; .05) indicates that the row and column categorical variables are dependent.</div>"

        tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
        self.desc_tabs.addTab(tv, f"Crosstabs ({timestamp})")
        self.desc_tabs.setCurrentIndex(self.desc_tabs.count() - 1)


    # ==========================================
    # MODULE 2.5: ITEM ANALYSIS & CVI
    # ==========================================
    def init_item_analysis_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.add_info_box(cl, "<b>Item Analysis & CVI:</b> Evaluate psychometric properties of survey items. Computes Item Difficulty (Mean), Item Discrimination (Item-Rest Correlation), and Alternate Weight (Alpha if Item Deleted).")


        ig = QGroupBox("Classical Item Analysis")
        ig.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        igl = QVBoxLayout(ig)
        
        self.item_vars = QListWidget(); self.item_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.item_vars.setMinimumHeight(120)
        self.setup_list_selection(igl, "Select Items to Analyze:", self.item_vars, "Item Analysis")
        
        ik_vbox = QVBoxLayout()
        ik_vbox.addSpacing(10) 
        lbl_key = QLabel("<b>Scoring Key (Optional):</b><br>Type the correct answer (e.g., <b>1</b> or <b>A</b>).<br>Matches are recorded as 1 (Correct), others as 0 (Incorrect).")
        lbl_key.setWordWrap(True)
        lbl_key.setMinimumWidth(10) # Fluid Layout Trick
        ik_vbox.addWidget(lbl_key)
        
        self.item_key_input = QLineEdit()
        self.item_key_input.setPlaceholderText("Leave blank to use raw scores")
        self.item_key_input.setMaximumWidth(250) 
        ik_vbox.addWidget(self.item_key_input)
        
        igl.addLayout(ik_vbox)

        btn_ia = QPushButton("▶ Run Item Analysis"); btn_ia.clicked.connect(self.run_item_analysis)
        igl.addWidget(btn_ia)
        cl.addWidget(ig)
        
        cg = QGroupBox("Content Validity Index (I-CVI) Calculator")
        cg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        cgl = QVBoxLayout(cg)
        
        self.item_cvi_vars = QListWidget(); self.item_cvi_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.item_cvi_vars.setMinimumHeight(120)
        self.setup_list_selection(cgl, "Select Expert Rating Columns:", self.item_cvi_vars, "I-CVI Raters")
        
        self.cvi_thresh = QDoubleSpinBox(); self.cvi_thresh.setRange(0.01, 10.0); self.cvi_thresh.setValue(0.5)
        ch = QFormLayout()
        ch.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        ch.addRow("Threshold to consider an item 'Relevant' (e.g. > 0):", self.cvi_thresh)
        cgl.addLayout(ch)
        btn_cvi = QPushButton("▶ Calculate I-CVI"); btn_cvi.clicked.connect(self.run_cvi)
        cgl.addWidget(btn_cvi)
        cl.addWidget(cg)

        tab, self.ia_tabs = self.create_split_module(cw, bank=self.build_bank_panel())
        self.tabs.addWidget(tab)

    def run_item_analysis(self):
        if self.df is None: return
        selected = [item.text() for item in self.item_vars.selectedItems()]
        if len(selected) < 2: return
        
        try:
            raw_data = self.df[selected].dropna()
            key_str = self.item_key_input.text().strip()
            
            out = self.get_apa_css() + f"<h2>Item Analysis Statistics</h2>"
            out += self.build_missing_data_note(self.df, selected, len(raw_data))
            
            if key_str:
                try: key_val = float(key_str)
                except ValueError: key_val = key_str
                
                data = (raw_data == key_val).astype(int)
                out += f"<p><b>Scoring Key Applied:</b> Responses matching '<b>{key_str}</b>' were recoded to 1 (Correct). All other responses were recoded to 0 (Incorrect).</p>"
            else:
                data = raw_data
                
            total_score = data.sum(axis=1)
            overall_alpha = calc_cronbach_alpha(data)
            overall_omega, omega_loadings = calc_mcdonalds_omega(data)
            
            out += "<div style='display:flex; gap:20px; margin:14px 0;'>"
            out += f"<div style='flex:1; background:#EEF2FF; border:2px solid #4F46E5; border-radius:10px; padding:14px; text-align:center;'>"
            out += f"<div style='font-size:13px; color:#4338CA; font-weight:bold;'>CRONBACH'S α</div>"
            out += f"<div style='font-size:30px; font-weight:900; color:#4338CA;'>{self.fmt(overall_alpha)}</div></div>"
            out += f"<div style='flex:1; background:#ECFDF5; border:2px solid #059669; border-radius:10px; padding:14px; text-align:center;'>"
            out += f"<div style='font-size:13px; color:#047857; font-weight:bold;'>McDONALD'S ω (OMEGA)</div>"
            if not np.isnan(overall_omega):
                out += f"<div style='font-size:30px; font-weight:900; color:#047857;'>{self.fmt(overall_omega)}</div></div>"
            else:
                out += f"<div style='font-size:16px; font-weight:bold; color:#9CA3AF;'>N/A (needs ≥3 items &amp; factor_analyzer)</div></div>"
            out += "</div>"
            if not np.isnan(overall_omega):
                out += "<p style='font-size:12px; color:#6B7280;'><i>ω estimated from a single-factor model (McDonald, 1999); unlike α it does not assume equal item loadings, so it is generally the more accurate reliability estimate when loadings vary.</i></p>"
            
            out += "<table class='apa'><tr><th>Item</th><th>Mean (Difficulty)</th><th>SD</th><th>Item-Rest Correlation (Discrimination)</th><th>Cronbach's α if Deleted</th><th>Factor Loading (λ)</th></tr>"
            
            for col in selected:
                item_mean = data[col].mean()
                item_sd = data[col].std()
                rest_score = total_score - data[col]
                
                if data[col].std() == 0 or rest_score.std() == 0:
                    disc_r = np.nan
                else:
                    try: disc_r, _ = stats.pearsonr(data[col], rest_score)
                    except: disc_r = np.nan
                
                del_data = data.drop(columns=[col])
                alpha_del = calc_cronbach_alpha(del_data)
                load_str = self.fmt(omega_loadings[col]) if omega_loadings is not None and col in omega_loadings.index else "—"
                
                out += f"<tr><td style='text-align:left;'><b>{col}</b></td><td>{self.fmt(item_mean)}</td><td>{self.fmt(item_sd)}</td><td>{self.fmt(disc_r)}</td><td>{self.fmt(alpha_del)}</td><td>{load_str}</td></tr>"
                
            out += "</table><div class='interpret'><i>Note.</i> <b>Guidelines (DeVellis, 2016):</b><br><b>Item Difficulty (Mean):</b> Represents the average score or proportion of endorsement. For 0/1 coded data, this is the proportion of correct answers (p-value).<br><b>Discrimination:</b> Item-rest correlations &lt; .20 indicate the item may not be measuring the same construct.<br><b>Alternate Weight (α if deleted):</b> If this value is higher than the Overall α, removing the item improves scale reliability.<br><b>Reliability thresholds (Hair et al., 2019):</b> α and ω ≥ .70 = acceptable, ≥ .80 = good, ≥ .90 = excellent.</div>"
            
            apa_block = self.build_apa_writeup("Reliability", {
                "k": len(selected), "alpha": overall_alpha, "omega": overall_omega
            })
            out += apa_block
            
            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.ia_tabs.addTab(tv, f"Item Stats ({timestamp})")
            self.ia_tabs.setCurrentIndex(self.ia_tabs.count() - 1)
        except Exception as e:
            self.ia_tabs.addTab(QTextEdit(str(e)), "Error")

    def run_cvi(self):
        if self.df is None: return
        raters = [item.text() for item in self.item_cvi_vars.selectedItems()]
        if not raters: return
        
        try:
            data = self.df[raters].dropna()
            threshold = self.cvi_thresh.value()
            
            out = self.get_apa_css() + "<h2>Content Validity Index (I-CVI)</h2>"
            out += f"<p><b>Number of Experts/Raters:</b> {len(raters)}</p>"
            out += "<table class='apa'><tr><th>Row Index (Item)</th><th>Relevant Ratings Count</th><th>I-CVI</th><th>Status</th></tr>"
            
            sum_cvi = 0
            ua_count = 0
            for idx in range(len(data)):
                row_ratings = data.iloc[idx]
                relevant_count = (pd.to_numeric(row_ratings, errors='coerce') > threshold).sum()
                i_cvi = relevant_count / len(raters)
                
                sum_cvi += i_cvi
                if relevant_count == len(raters): ua_count += 1
                
                status = "<span class='sig'>Excellent</span>" if i_cvi >= 0.78 else ("<span class='warn'>Revise/Reject</span>" if i_cvi < 0.70 else "Acceptable")
                out += f"<tr><td>Item Row {idx+1}</td><td>{relevant_count}</td><td>{self.fmt(i_cvi)}</td><td>{status}</td></tr>"
                
            s_cvi_ave = sum_cvi / len(data)
            s_cvi_ua = ua_count / len(data)
            
            out += "</table>"
            out += f"<p><b>Scale-level CVI (S-CVI/Ave):</b> {self.fmt(s_cvi_ave)}</p>"
            out += f"<p><b>Scale-level CVI Universal Agreement (S-CVI/UA):</b> {self.fmt(s_cvi_ua)}</p>"
            out += "<div class='interpret'><i>Note.</i> <b>Guidelines (Lynn, 1986; Polit & Beck, 2006):</b> I-CVI should be 1.00 when there are 3-5 experts, and ≥ .78 when there are 6 or more experts to be considered excellent evidence of content validity. S-CVI/Ave ≥ .90 is recommended for scale-level validation.</div>"

            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.ia_tabs.addTab(tv, f"I-CVI ({timestamp})")
            self.ia_tabs.setCurrentIndex(self.ia_tabs.count() - 1)
        except Exception as e:
            self.ia_tabs.addTab(QTextEdit(str(e)), "Error")


    # ==========================================
    # MODULE 3: CORRELATION
    # ==========================================
    def init_correlation_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.add_info_box(cl, "<b>Correlation:</b> Used to evaluate the strength and direction of a linear relationship between two numeric variables. It tells you if they move together (positive) or in opposite directions (negative).")


        cl.addWidget(QLabel("Method:"))
        self.corr_method = QComboBox()
        self.corr_method.addItems(["Pearson (Parametric)", "Spearman (Non-Parametric)"])
        cl.addWidget(self.corr_method)

        cl.addWidget(QLabel("Missing Value Handling:"))
        self.corr_missing = QComboBox()
        self.corr_missing.addItems(["Listwise Deletion (exclude case if ANY variable missing)", "Pairwise Deletion (use all available data per pair)"])
        self.corr_missing.setToolTip(
            "Listwise: every correlation in the matrix uses the same N (rows with any missing value across "
            "all selected variables are dropped). Conservative and guarantees a consistent, valid covariance "
            "structure, but discards usable data.\n\n"
            "Pairwise: each correlation uses all cases with non-missing data for that specific pair, so N can "
            "differ cell-to-cell. Uses more of your data, but with enough missingness the resulting matrix is "
            "not guaranteed to be positive semi-definite (a mathematical requirement for some downstream "
            "techniques, e.g. factor analysis or SEM run on a correlation matrix directly)."
        )
        cl.addWidget(self.corr_missing)

        cl.addWidget(QLabel("Multiple Comparisons Correction:"))
        self.corr_correction = QComboBox()
        self.corr_correction.addItems(["None (raw p-values)", "Bonferroni", "Benjamini-Hochberg (FDR)"])
        self.corr_correction.setToolTip(
            "When testing many correlations at once, each individual test still carries a 5% false-positive "
            "risk, so the chance that AT LEAST ONE comes out 'significant' purely by chance grows quickly with "
            "the number of variables. Bonferroni is the strictest, simplest correction (controls the chance of "
            "ANY false positive); Benjamini-Hochberg FDR is less conservative (controls the expected PROPORTION "
            "of false positives among significant results) and retains more power -- a common practical choice "
            "when screening many correlations exploratively."
        )
        cl.addWidget(self.corr_correction)
        
        self.corr_vars = QListWidget()
        self.corr_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.corr_vars.setMinimumHeight(150)
        self.setup_list_selection(cl, "Select Variables:", self.corr_vars, "Correlation Variables")
        
        btn = QPushButton("▶ Run Correlation")
        btn.clicked.connect(self.run_correlation)
        cl.addWidget(btn)
        
        tab, self.corr_tabs = self.create_split_module(cw, bank=self.build_bank_panel(numeric_only=True))
        self.tabs.addWidget(tab)

    def run_correlation(self):
        if self.df is None: return
        selected = [item.text() for item in self.corr_vars.selectedItems()]
        if len(selected) < 2: return
        
        method = "pearson" if "Pearson" in self.corr_method.currentText() else "spearman"
        is_pairwise = "Pairwise" in self.corr_missing.currentText()
        
        out = self.get_apa_css()

        r_mat = pd.DataFrame(np.eye(len(selected)), index=selected, columns=selected)
        p_mat = pd.DataFrame(np.zeros((len(selected), len(selected))), index=selected, columns=selected)
        n_mat = pd.DataFrame(np.zeros((len(selected), len(selected)), dtype=int), index=selected, columns=selected)

        if is_pairwise:
            # Each cell uses its own maximal available pair -- N can differ cell to cell.
            # No single shared "clean data" exists, so the missing-data note is computed
            # per-variable instead of as one shared listwise drop.
            out += self.build_missing_data_note(self.df, selected, len(self.df[selected].dropna()))
            out += ("<p style='font-size:12.5px; color:#6B7280;'><i>Pairwise deletion in use: the figures above "
                    "describe what listwise deletion would have discarded. Each correlation below instead uses "
                    "its own maximum available N (shown in parentheses), which may differ cell to cell.</i></p>")
        else:
            data = self.df[selected].dropna()
            out += self.build_missing_data_note(self.df, selected, len(data))

        out += f"<h2>{method.capitalize()} Correlation Matrix ({'Pairwise' if is_pairwise else 'Listwise'} Deletion)</h2>"

        # --- Pass 1: compute every unique pairwise correlation first (without rendering),
        # so a multiple-comparisons correction can be applied across the full set of
        # p-values before anything is drawn -- the correction needs to know about every
        # test in the family at once, not just the ones rendered so far. ---
        cell_results = {}  # (var1, var2) unordered pair -> (r, p, n)
        for i, var1 in enumerate(selected):
            for j, var2 in enumerate(selected):
                if i >= j:
                    continue
                if is_pairwise:
                    pair_data = self.df[[var1, var2]].dropna()
                else:
                    pair_data = data[[var1, var2]]
                n_pair = len(pair_data)
                if n_pair < 3:
                    r, p = np.nan, np.nan
                else:
                    r, p = stats.pearsonr(pair_data[var1], pair_data[var2]) if method == "pearson" else stats.spearmanr(pair_data[var1], pair_data[var2])
                cell_results[(var1, var2)] = (r, p, n_pair)
                r_mat.loc[var1, var2] = r; r_mat.loc[var2, var1] = r
                p_mat.loc[var1, var2] = p; p_mat.loc[var2, var1] = p
                n_mat.loc[var1, var2] = n_pair; n_mat.loc[var2, var1] = n_pair

        correction_choice = self.corr_correction.currentText()
        corrected_lookup = {}
        if "None" not in correction_choice:
            pair_keys = [k for k, v in cell_results.items() if not pd.isna(v[1])]
            raw_ps = [cell_results[k][1] for k in pair_keys]
            if raw_ps:
                try:
                    method_code = 'bonferroni' if 'Bonferroni' in correction_choice else 'fdr_bh'
                    _, p_corr, _, _ = multipletests(raw_ps, alpha=0.05, method=method_code)
                    corrected_lookup = dict(zip(pair_keys, p_corr))
                except Exception:
                    pass

        strongest = None  # (abs_r, var1, var2, r, p, n)
        for k, (r, p, n_pair) in cell_results.items():
            if not pd.isna(r) and (strongest is None or abs(r) > strongest[0]):
                strongest = (abs(r), k[0], k[1], r, p, n_pair)

        # --- Pass 2: render the table using the already-computed matrix + corrections ---
        out += "<table class='apa'><tr><th>Variable</th>"
        for col in selected: out += f"<th>{col}</th>"
        out += "</tr>"
        
        for i, var1 in enumerate(selected):
            out += f"<tr><td style='text-align:left;'><b>{var1}</b></td>"
            for j, var2 in enumerate(selected):
                if i == j: 
                    out += "<td>-</td>"
                else:
                    key = (var1, var2) if (var1, var2) in cell_results else (var2, var1)
                    r, p, n_pair = cell_results.get(key, (np.nan, np.nan, 0))
                    p_corr = corrected_lookup.get(key, None)
                    stars = "" if pd.isna(p) else ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
                    n_suffix = f"<br><span style='font-size:11px;color:#9CA3AF;'>(n={n_pair})</span>" if is_pairwise else ""

                    sig_for_color = (not pd.isna(p_corr)) if p_corr is not None else (not pd.isna(p) and p < 0.05)
                    if p_corr is not None:
                        sig_for_color = (not pd.isna(p_corr)) and p_corr < 0.05

                    p_line = f"(p = {self.fmt(p, True)})"
                    if p_corr is not None:
                        p_line += f"<br>(corrected p = {self.fmt(p_corr, True)})"
                    
                    if sig_for_color:
                        val = f"<span class='sig'>{self.fmt(r)}{stars}</span><br><span style='font-size:12px;color:#6B7280;'>{p_line}</span>{n_suffix}"
                    else:
                        val = f"{self.fmt(r)}{stars}<br><span style='font-size:12px;color:#6B7280;'>{p_line}</span>{n_suffix}"
                    out += f"<td>{val}</td>"
            out += "</tr>"
            
        out += "</table><p style='font-size:13px;'><i>* p < .05, ** p < .01, *** p < .001 (uncorrected). Bolded values are significant after any correction applied below.</i></p>"
        out += "<div class='interpret'><i>Note.</i> <b>Interpretation (Cohen, 1988):</b> Effect sizes for correlations: <i>r</i> &approx; .10 (Small), .30 (Medium), .50 (Large). Significant values are bolded.</div>"
        if "None" not in correction_choice:
            n_tests = len([v for v in cell_results.values() if not pd.isna(v[1])])
            out += (
                f"<div style='background:#EFF6FF; border-left:4px solid #3B82F6; padding:8px 12px; margin:10px 0; "
                f"font-size:13px; color:#1E3A8A;'><b>{correction_choice} correction applied</b> across "
                f"{n_tests} pairwise tests. Bolding above now reflects the corrected p-value, not the raw one.</div>"
            )
        
        min_n_used = int(n_mat.values[~np.eye(len(selected), dtype=bool)].min()) if len(selected) > 1 else len(self.df)
        out += self.build_sample_size_warning(min_n_used, "correlation")
        
        if strongest is not None:
            _, v1, v2, r_s, p_s, n_s = strongest
            out += self.build_apa_writeup("Correlation", {
                "method": method.capitalize(), "var1": v1, "var2": v2,
                "r": r_s, "p": p_s, "n": n_s
            })
        
        tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.corr_tabs.addTab(tv, f"Correlation ({timestamp})")
        self.corr_tabs.setCurrentIndex(self.corr_tabs.count() - 1)

        # --- Modern diagonal correlation heatmap (lower triangle only) ---
        if MATPLOTLIB_AVAILABLE:
            try:
                heat_widget = self.build_correlation_heatmap(r_mat, p_mat, method)
                if heat_widget is not None:
                    self.corr_tabs.addTab(heat_widget, f"Heatmap ({timestamp})")
            except Exception:
                pass

        self.corr_tabs.setCurrentIndex(self.corr_tabs.count() - 1)

    def build_correlation_heatmap(self, r_mat, p_mat, method):
        """
        Builds a publication-ready, lower-triangle-only correlation heatmap (the
        upper triangle and diagonal are masked, matching the modern style popularized
        by seaborn's corrplot examples and JASP's correlation matrix plot). Significant
        cells (p < .05) are annotated with an asterisk. Returns a QWidget for the
        results tab, or None if too few variables are present.
        """
        n = r_mat.shape[0]
        if n < 2:
            return None
        labels = r_mat.columns.tolist()
        mask = np.triu(np.ones((n, n), dtype=bool), k=0)  # hide diagonal + upper triangle
        vals = r_mat.values.copy()
        vals_masked = np.ma.array(vals, mask=mask)

        fig = Figure(figsize=(max(5, n * 0.9), max(4, n * 0.8)))
        ax = fig.add_subplot(111)
        is_dark = self.is_dark_mode
        if is_dark:
            fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#1F2937')
        import matplotlib.cm as cm
        try:
            cmap = matplotlib.colormaps['RdBu_r']
        except Exception:
            cmap = cm.get_cmap('RdBu_r')
        cmap = cmap.copy()
        cmap.set_bad(color=(0, 0, 0, 0))  # masked cells fully transparent

        im = ax.imshow(vals_masked, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        tick_color = 'white' if is_dark else 'black'
        ax.tick_params(colors=tick_color)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    continue
                r_val = r_mat.values[i, j]
                p_val = p_mat.values[i, j]
                star = "*" if p_val < 0.05 else ""
                text_color = 'white' if abs(r_val) > 0.55 else ('#E5E7EB' if is_dark else '#111827')
                ax.text(j, i, f"{r_val:.2f}{star}", ha='center', va='center', fontsize=8.5, color=text_color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=tick_color)
        title_color = 'white' if is_dark else 'black'
        ax.set_title(f"{method.capitalize()} Correlation Heatmap", fontweight='bold', color=title_color, fontsize=13)
        fig.tight_layout()

        chart = self.make_zoomable_chart(fig)
        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)
        note = QLabel("Lower-triangle heatmap. Color intensity = correlation strength. * indicates p < .05.")
        note.setObjectName("InfoLabel")
        note.setWordWrap(True)
        wl.addWidget(note)
        wl.addWidget(chart)
        return wrapper



    # ==========================================
    # MODULE 4: COMPARE MEANS 
    # ==========================================
    def init_compare_means_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop) 
        
        self.add_info_box(cl, "<b>Compare Means (T-Tests):</b> Evaluate if there is a statistically significant difference between the means of two groups (Independent) or two time points/conditions (Paired).")


        self.r_indep = QRadioButton("Independent Samples"); self.r_paired = QRadioButton("Paired Samples"); self.r_indep.setChecked(True)
        tg = QGroupBox("Test Design"); tg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        tl = QVBoxLayout(tg); tl.addWidget(self.r_indep); tl.addWidget(self.r_paired); cl.addWidget(tg)
        
        self.r_param = QRadioButton("Parametric (T-Tests)"); self.r_nonparam = QRadioButton("Non-Parametric (U/W)"); self.r_param.setChecked(True)
        ag = QGroupBox("Assumption"); ag.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        al = QVBoxLayout(ag); al.addWidget(self.r_param); al.addWidget(self.r_nonparam)
        self.t_welch = QCheckBox("Use Welch's correction if Levene's test is significant")
        self.t_welch.setChecked(True)
        self.t_welch.setToolTip(
            "Welch's t-test (Welch, 1947) does not assume the two groups have equal variances, unlike the "
            "classic Student's t-test. When Levene's test below flags unequal variances, Welch's correction "
            "is the modern recommended default -- it costs almost nothing when variances ARE equal, but "
            "protects against an inflated Type I error rate when they aren't."
        )
        al.addWidget(self.t_welch)
        cl.addWidget(ag)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        self.t_dv = QComboBox()
        self.t_iv = QComboBox()
        self.enable_drag_drop_combo(self.t_dv)
        self.enable_drag_drop_combo(self.t_iv)
        form.addRow(QLabel("Dependent Variable:"), self.t_dv)
        form.addRow(QLabel("Grouping Variable:"), self.t_iv)
        cl.addLayout(form)
        
        btn = QPushButton("▶ Run Comparison")
        btn.setStyleSheet("margin-top: 15px;")
        btn.clicked.connect(self.run_compare)
        cl.addWidget(btn)
        cl.addStretch()
        
        tab, self.t_tabs = self.create_split_module(cw, bank=self.build_bank_panel())
        self.tabs.addWidget(tab)

    def run_compare(self):
        if self.df is None: return
        dv, iv = self.t_dv.currentText(), self.t_iv.currentText()
        if not dv or not iv: return
        
        out = self.get_apa_css() + f"<h2>Compare Means: {dv} by {iv}</h2>"
        raincloud_groups = None
        apa_payload = None
        
        if self.r_indep.isChecked():
            data = self.df[[dv, iv]].dropna()
            out += self.build_missing_data_note(self.df, [dv, iv], len(data))
            out += "<table class='apa'><tr><th>Group</th><th>N</th><th>Mean</th><th>SD</th><th>Statistic</th><th>p-value</th><th>Cohen's d</th></tr>"
            groups = data[iv].unique()
            if len(groups) != 2: 
                err_tv = QTextEdit("<p class='warn'>Error: Independent test requires exactly 2 groups in the Grouping Variable.</p>")
                self.t_tabs.addTab(err_tv, "Error")
                self.t_tabs.setCurrentIndex(self.t_tabs.count() - 1)
                return
            g1, g2 = data[data[iv] == groups[0]][dv], data[data[iv] == groups[1]][dv]
            
            s_pool = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
            cd = abs(g1.mean() - g2.mean()) / s_pool if s_pool > 0 else 0

            levene_stat, levene_p = (np.nan, np.nan)
            use_welch = False
            try:
                levene_stat, levene_p = stats.levene(g1, g2, center='median')
                use_welch = self.t_welch.isChecked() and (not pd.isna(levene_p)) and levene_p < 0.05
            except Exception:
                pass
            
            if self.r_param.isChecked():
                stat, p = stats.ttest_ind(g1, g2, equal_var=not use_welch)
                s_name = f"t = {self.fmt(stat)}"
                if use_welch:
                    # Welch-Satterthwaite approximate df is non-integer; scipy computes it
                    # internally but doesn't return it directly, so derive it the same way.
                    v1, v2, n1, n2 = g1.var(ddof=1), g2.var(ddof=1), len(g1), len(g2)
                    df_val = ((v1/n1 + v2/n2)**2) / (((v1/n1)**2)/(n1-1) + ((v2/n2)**2)/(n2-1))
                else:
                    df_val = len(g1) + len(g2) - 2
            else:
                stat, p = stats.mannwhitneyu(g1, g2); s_name = f"U = {self.fmt(stat)}"
                df_val = None
            
            bp = f"<span class='sig'>{self.fmt(p, True)}</span>" if p < 0.05 else self.fmt(p, True)
            out += f"<tr><td>{groups[0]}</td><td>{len(g1)}</td><td>{self.fmt(g1.mean())}</td><td>{self.fmt(g1.std())}</td><td rowspan='2' style='vertical-align:middle;'>{s_name}</td><td rowspan='2' style='vertical-align:middle;'>{bp}</td><td rowspan='2' style='vertical-align:middle;'>{self.fmt(cd)}</td></tr>"
            out += f"<tr><td>{groups[1]}</td><td>{len(g2)}</td><td>{self.fmt(g2.mean())}</td><td>{self.fmt(g2.std())}</td></tr>"

            raincloud_groups = {str(groups[0]): g1.values, str(groups[1]): g2.values}
            apa_payload = {
                "dv": dv, "iv": iv, "design": "independent-samples", "stat_name": s_name.split(" = ")[0],
                "stat": stat, "df_val": df_val, "p": p, "d_val": cd,
                "g1_name": str(groups[0]), "g1_mean": g1.mean(), "g1_sd": g1.std(),
                "g2_name": str(groups[1]), "g2_mean": g2.mean(), "g2_sd": g2.std(),
                "welch_used": use_welch,
            }
            out += self.build_sample_size_warning(min(len(g1), len(g2)), "ttest")

            if not pd.isna(levene_p):
                lev_status = "Violated" if levene_p < 0.05 else "Pass"
                lev_color = "#EF4444" if levene_p < 0.05 else "#10B981"
                welch_note = ""
                if levene_p < 0.05:
                    welch_note = (" Welch's correction was automatically applied above." if use_welch
                                   else " Consider enabling Welch's correction (checkbox on the left) since this assumption is violated.")
                out += (
                    f"<div style='background:#F9FAFB; border-left:4px solid {lev_color}; padding:8px 12px; "
                    f"margin:10px 0; font-size:13px;'>"
                    f"<b>Levene's Test for Equality of Variances:</b> W = {self.fmt(levene_stat)}, "
                    f"{self.apa_p(levene_p)} &mdash; <span style='color:{lev_color}; font-weight:bold;'>{lev_status}</span> "
                    f"(p &gt; .05 indicates equal variances, the assumption of the classic Student's t-test).{welch_note}</div>"
                )
        else:
            # IMPORTANT: paired-samples designs require row-wise correspondence between
            # dv and iv (e.g. each row is one subject's pre/post pair). Dropping NAs from
            # each column independently (the previous behavior) can silently misalign
            # subjects whenever the two columns have missing values in different rows --
            # scipy's ttest_rel/wilcoxon pair by position, not by index, so a coincidental
            # length match would produce a wrong result with no error raised. A single
            # paired dropna() on both columns together is required for correctness.
            paired_data = self.df[[dv, iv]].dropna()
            out += self.build_missing_data_note(self.df, [dv, iv], len(paired_data))
            out += "<table class='apa'><tr><th>Group</th><th>N</th><th>Mean</th><th>SD</th><th>Statistic</th><th>p-value</th><th>Cohen's d</th></tr>"
            g1, g2 = paired_data[dv], paired_data[iv]
            cd = abs(g1.mean() - g2.mean()) / (g1 - g2).std() if (g1 - g2).std() > 0 else 0
            if self.r_param.isChecked():
                stat, p = stats.ttest_rel(g1, g2); s_name = f"t = {self.fmt(stat)}"
                df_val = min(len(g1), len(g2)) - 1
            else:
                stat, p = stats.wilcoxon(g1, g2); s_name = f"W = {self.fmt(stat)}"
                df_val = None
            
            bp = f"<span class='sig'>{self.fmt(p, True)}</span>" if p < 0.05 else self.fmt(p, True)
            out += f"<tr><td>{dv}</td><td>{len(g1)}</td><td>{self.fmt(g1.mean())}</td><td>{self.fmt(g1.std())}</td><td rowspan='2' style='vertical-align:middle;'>{s_name}</td><td rowspan='2' style='vertical-align:middle;'>{bp}</td><td rowspan='2' style='vertical-align:middle;'>{self.fmt(cd)}</td></tr>"
            out += f"<tr><td>{iv}</td><td>{len(g2)}</td><td>{self.fmt(g2.mean())}</td><td>{self.fmt(g2.std())}</td></tr>"

            apa_payload = {
                "dv": dv, "iv": iv, "design": "paired-samples", "stat_name": s_name.split(" = ")[0],
                "stat": stat, "df_val": df_val, "p": p, "d_val": cd,
                "g1_name": dv, "g1_mean": g1.mean(), "g1_sd": g1.std(),
                "g2_name": iv, "g2_mean": g2.mean(), "g2_sd": g2.std(),
            }
            out += self.build_sample_size_warning(len(g1), "ttest")
            
        out += "</table><div class='interpret'><i>Note.</i> <b>Interpretation (Field, 2013):</b> A p-value &lt; .05 indicates that the mean difference is statistically significant. <b>Cohen (1988) Effect Size:</b> 0.2 (Small), 0.5 (Medium), 0.8 (Large).</div>"
        if apa_payload is not None:
            out += self.build_apa_writeup("TTest", apa_payload)
        tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.t_tabs.addTab(tv, f"T-Test ({timestamp})")
        self.t_tabs.setCurrentIndex(self.t_tabs.count() - 1)

        # Raincloud plot is hard-coded to trigger alongside Independent T-Tests.
        if raincloud_groups is not None:
            try:
                rc_widget = self.build_raincloud_plot(raincloud_groups, dv, title=f"{dv} by {iv}")
                if rc_widget is not None:
                    self.t_tabs.addTab(rc_widget, f"Raincloud ({timestamp})")
            except Exception:
                pass
        self.t_tabs.setCurrentIndex(self.t_tabs.count() - 1)


    # ==========================================
    # MODULE 5: ANALYSIS OF VARIANCE (ANOVA/MANOVA)
    # ==========================================
    def init_anova_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.add_info_box(cl, "<b>Analysis of Variance:</b> Test for mean differences across 3 or more groups. Use 'Between-Subjects' when each group is a different set of people; use 'Repeated Measures' when the same people were measured at multiple time points or conditions.")

        # Vertical toggle instead of side-by-side tabs: choosing "Between-Subjects" or
        # "Repeated Measures" switches which configuration panel is shown below, stacked
        # in the same narrow column rather than needing two tab labels to fit side by
        # side -- the window never needs to be widened to read either option.
        design_toggle_box = QGroupBox("Design")
        design_toggle_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        design_toggle_layout = QVBoxLayout(design_toggle_box)
        self.anova_design_bw = QRadioButton("Between-Subjects (ANOVA / MANOVA)")
        self.anova_design_rm = QRadioButton("Repeated Measures (RM-ANOVA / Friedman)")
        self.anova_design_bw.setChecked(True)
        design_toggle_layout.addWidget(self.anova_design_bw)
        design_toggle_layout.addWidget(self.anova_design_rm)
        cl.addWidget(design_toggle_box)

        anova_mode_stack = QStackedWidget()

        # --- Between-Subjects (Factorial) ANOVA / MANOVA / Kruskal-Wallis ---
        bw_tab = QWidget()
        bw_cl = QVBoxLayout(bw_tab)
        bw_cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        bw_cl.setContentsMargins(10, 10, 10, 10)

        ag = QGroupBox("Configuration")
        ag.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        agl = QVBoxLayout(ag)
        agl.setSpacing(5)

        self.anova_dv = QListWidget(); self.anova_dv.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.anova_dv.setMinimumHeight(80)
        self.setup_list_selection(agl, "Dependent Variables (Select 1 for ANOVA, >1 for MANOVA):", self.anova_dv, "Dependent Variables")
        
        self.anova_iv = QListWidget(); self.anova_iv.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.anova_iv.setMinimumHeight(80)
        self.setup_list_selection(agl, "Fixed Factors (Categorical):", self.anova_iv, "Fixed Factors")
        
        self.anova_covar = QListWidget(); self.anova_covar.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.anova_covar.setMinimumHeight(80)
        self.setup_list_selection(agl, "Covariates (Optional Metric):", self.anova_covar, "Covariates")
        
        self.anova_interact = QCheckBox("Full Factorial (Include Interaction Effects for Factors)")
        agl.addWidget(self.anova_interact)

        self.anova_kw = QCheckBox("Also run Kruskal-Wallis (non-parametric one-way alternative)")
        self.anova_kw.setToolTip(
            "Kruskal-Wallis H test (Kruskal & Wallis, 1952) is the non-parametric equivalent of one-way ANOVA. "
            "It does not assume normally distributed residuals or equal variances, and works on ranks instead of "
            "raw scores. Use it when Levene's test or the residual normality check flags a violation, or when "
            "your dependent variable is ordinal rather than truly continuous. Only available for a single DV and "
            "a single factor (the classic one-way design)."
        )
        agl.addWidget(self.anova_kw)

        self.anova_show_desc = QCheckBox("Include Descriptive Statistics (Mean, SD, N per group)")
        self.anova_show_desc.setChecked(True)
        agl.addWidget(self.anova_show_desc)
        
        btn = QPushButton("▶ Run Analysis of Variance")
        btn.setStyleSheet("margin-top: 10px;")
        btn.clicked.connect(self.run_anova)
        agl.addWidget(btn)
        bw_cl.addWidget(ag)
        bw_cl.addStretch()

        bw_scroll = QScrollArea()
        bw_scroll.setWidgetResizable(True)
        bw_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bw_scroll.setFrameShape(QFrame.Shape.NoFrame)
        bw_scroll.setWidget(bw_tab)
        anova_mode_stack.addWidget(bw_scroll)

        # --- Repeated Measures: RM-ANOVA (parametric) + Friedman (non-parametric) ---
        rm_tab = QWidget()
        rm_cl = QVBoxLayout(rm_tab)
        rm_cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        rm_cl.setContentsMargins(10, 10, 10, 10)

        self.add_info_box(rm_cl, "<b>Repeated Measures:</b> Select 3 or more columns representing the SAME measure taken at different time points or conditions on the SAME subjects (e.g. Pretest, Midtest, Posttest). For exactly 2 time points, use the Paired T-Test in Compare Means instead.")

        rmg = QGroupBox("Repeated Measures Configuration")
        rmg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        rmgl = QVBoxLayout(rmg)
        rmgl.setSpacing(5)

        self.rm_vars = QListWidget(); self.rm_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.rm_vars.setMinimumHeight(120)
        self.setup_list_selection(rmgl, "Repeated Measures (select 3+, in time order):", self.rm_vars, "Repeated Measures")

        self.rm_test_type = QComboBox()
        self.rm_test_type.addItems([
            "Repeated-Measures ANOVA (Parametric)",
            "Friedman Test (Non-Parametric)",
            "Both",
        ])
        self.rm_test_type.setToolTip(
            "RM-ANOVA assumes the differences between each pair of conditions have roughly equal variance "
            "(sphericity) -- this implementation does not test for or correct sphericity violations. "
            "Friedman's test (Friedman, 1937) makes no such assumption, working on ranks instead, and is the "
            "safer default with ordinal data, non-normal residuals, or unequal within-subject variances."
        )
        rmgl.addWidget(QLabel("Test:"))
        rmgl.addWidget(self.rm_test_type)

        self.rm_show_desc = QCheckBox("Include Descriptive Statistics (Mean, SD, N per timepoint)")
        self.rm_show_desc.setChecked(True)
        rmgl.addWidget(self.rm_show_desc)

        btn_rm = QPushButton("▶ Run Repeated Measures Test")
        btn_rm.setStyleSheet("margin-top: 10px;")
        btn_rm.clicked.connect(self.run_repeated_measures)
        rmgl.addWidget(btn_rm)
        rm_cl.addWidget(rmg)
        rm_cl.addStretch()

        rm_scroll = QScrollArea()
        rm_scroll.setWidgetResizable(True)
        rm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rm_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rm_scroll.setWidget(rm_tab)
        anova_mode_stack.addWidget(rm_scroll)

        self.anova_design_bw.toggled.connect(lambda checked: anova_mode_stack.setCurrentIndex(0) if checked else None)
        self.anova_design_rm.toggled.connect(lambda checked: anova_mode_stack.setCurrentIndex(1) if checked else None)

        cl.addWidget(anova_mode_stack)
        
        tab, self.anova_tabs = self.create_split_module(cw, bank=self.build_bank_panel())
        self.tabs.addWidget(tab)

    def run_anova(self):
        if self.df is None: return
        dvs = [i.text() for i in self.anova_dv.selectedItems()]
        ivs = [i.text() for i in self.anova_iv.selectedItems()]
        covs = [i.text() for i in self.anova_covar.selectedItems()]
        if not dvs or not ivs: return
        
        try:
            factor_term = " * ".join([f"C({iv})" for iv in ivs]) if self.anova_interact.isChecked() else " + ".join([f"C({iv})" for iv in ivs])
            cov_term = " + " + " + ".join(covs) if covs else ""
            
            data_sub = self.df[dvs + ivs + covs].dropna()
            out = self.get_apa_css()
            out += self.build_missing_data_note(self.df, dvs + ivs + covs, len(data_sub))
            timestamp = datetime.now().strftime("%H:%M:%S")

            if self.anova_show_desc.isChecked():
                out += self.build_group_descriptives_table(data_sub, dvs, ivs[0])

            if len(dvs) == 1:
                dv = dvs[0]
                if self.anova_interact.isChecked():
                    # Type III SS requires sum (effects) coding so each main-effect
                    # coefficient is estimated at the grand mean, matching SPSS GLM.
                    # Dummy/treatment coding produces incorrect Type III SS for
                    # unbalanced designs and must not be used here.
                    sum_coded_ivs = [f"C({iv}, Sum)" for iv in ivs]
                    factor_term_sc = " * ".join(sum_coded_ivs)
                    cov_term_sc = " + " + " + ".join(covs) if covs else ""
                    formula = f"{dv} ~ {factor_term_sc}{cov_term_sc}"
                    model = smf.ols(formula, data=data_sub).fit()
                    aov = sm.stats.anova_lm(model, typ=3)
                    aov = aov.drop("Intercept", errors="ignore")
                    ss_type_label = "Type III"
                else:
                    formula = f"{dv} ~ {factor_term}{cov_term}"
                    model = smf.ols(formula, data=data_sub).fit()
                    aov = sm.stats.anova_lm(model, typ=2)
                    ss_type_label = "Type II"
                
                if 'Residual' in aov.index:
                    aov['Partial η²'] = aov['sum_sq'] / (aov['sum_sq'] + aov.loc['Residual', 'sum_sq'])
                    aov.loc['Residual', 'Partial η²'] = np.nan
                
                out += f"<h2>ANOVA / ANCOVA Results ({ss_type_label} Sums of Squares)</h2>"
                out += f"<p><b>Dependent Variable Analyzed:</b> {dv}</p>"
                out += "<table class='apa'><tr><th>Source</th><th>Sum of Sq.</th><th>df</th><th>F-value</th><th>p-value</th><th>Partial η²</th></tr>"
                for row in aov.itertuples():
                    source = row.Index
                    ss = self.fmt(row.sum_sq)
                    df_v = self.fmt(row.df)
                    f_v = self.fmt(row.F) if hasattr(row, 'F') and not pd.isna(row.F) else ""
                    p_v = row._4 if hasattr(row, '_4') else np.nan
                    
                    if pd.isna(p_v): p_str = ""
                    elif p_v < 0.05: p_str = f"<span class='sig'>{self.fmt(p_v, True)}</span>"
                    else: p_str = self.fmt(p_v, True)
                    
                    eta = self.fmt(getattr(row, '_5', np.nan)) if hasattr(row, '_5') else ""
                    out += f"<tr><td style='text-align:left;'>{source}</td><td>{ss}</td><td>{df_v}</td><td>{f_v}</td><td>{p_str}</td><td>{eta}</td></tr>"
                
                out += "</table><div class='interpret'><i>Note.</i> <b>Interpretation (Field, 2013):</b> If p &lt; .05, the factor (or interaction) has a significant effect. Partial η² is the effect size (0.01=Small, 0.06=Medium, 0.14=Large).</div>"
                
                # APA write-up for the first factor's main effect (covers the common one-way case cleanly)
                first_iv = ivs[0]
                main_effect_row = None
                for row_idx in aov.index:
                    if row_idx == f"C({first_iv})":
                        main_effect_row = aov.loc[row_idx]
                        break
                if main_effect_row is not None and 'Residual' in aov.index:
                    out += self.build_apa_writeup("ANOVA", {
                        "dv": dv, "iv_term": first_iv,
                        "f_val": main_effect_row.get('F', np.nan),
                        "df1": int(main_effect_row['df']) if not pd.isna(main_effect_row['df']) else "—",
                        "df2": int(aov.loc['Residual', 'df']) if not pd.isna(aov.loc['Residual', 'df']) else "—",
                        "p": main_effect_row.get('PR(>F)', np.nan),
                        "eta2": main_effect_row.get('Partial η²', np.nan),
                    })

                min_group_n = data_sub.groupby(first_iv).size().min() if first_iv in data_sub.columns else len(data_sub)
                out += self.build_sample_size_warning(int(min_group_n), "anova")

                # Levene's test for homogeneity of variance across the first factor's groups
                # -- a standard prerequisite check for ANOVA, alongside the post-hoc tests below.
                try:
                    lev_groups = [g[dv].values for _, g in data_sub.groupby(first_iv) if len(g) > 0]
                    if len(lev_groups) >= 2:
                        levene_stat, levene_p = stats.levene(*lev_groups, center='median')
                        lev_status = "Violated" if levene_p < 0.05 else "Pass"
                        lev_color = "#EF4444" if levene_p < 0.05 else "#10B981"
                        out += (
                            f"<div style='background:#F9FAFB; border-left:4px solid {lev_color}; padding:8px 12px; "
                            f"margin:10px 0; font-size:13px;'>"
                            f"<b>Levene's Test for Equality of Variances</b> (across <i>{first_iv}</i> groups): "
                            f"W = {self.fmt(levene_stat)}, {self.apa_p(levene_p)} &mdash; "
                            f"<span style='color:{lev_color}; font-weight:bold;'>{lev_status}</span> "
                            f"(p &gt; .05 indicates the groups have approximately equal variances, the assumption "
                            f"underlying the standard F-test used above). If violated, consider checking the "
                            f"\"Also run Kruskal-Wallis\" option (left panel) for a non-parametric alternative.</div>"
                        )
                except Exception:
                    pass

                # Kruskal-Wallis (non-parametric one-way alternative), only meaningful for the
                # classic single-DV, single-factor design -- skipped for MANOVA/factorial designs
                # where there's no single well-defined non-parametric equivalent here.
                if self.anova_kw.isChecked() and len(ivs) == 1 and not covs:
                    try:
                        kw_groups = [g[dv].values for _, g in data_sub.groupby(first_iv) if len(g) > 0]
                        if len(kw_groups) >= 2:
                            kw_stat, kw_p = stats.kruskal(*kw_groups)
                            kw_sig = f"<span class='sig'>{self.apa_p(kw_p)}</span>" if kw_p < 0.05 else self.apa_p(kw_p)
                            out += "<h2>Kruskal-Wallis Test (Non-Parametric)</h2>"
                            out += (
                                f"<p>H({len(kw_groups)-1}) = {self.fmt(kw_stat)}, {kw_sig}</p>"
                                f"<div class='interpret'><i>Note.</i> The Kruskal-Wallis H test (Kruskal &amp; Wallis, 1952) "
                                f"compares the ranked distributions of <i>{dv}</i> across <i>{first_iv}</i> groups without "
                                f"assuming normality or equal variances. A significant result (p &lt; .05) indicates at "
                                f"least one group's distribution differs; consider Dunn's test (not yet implemented here) "
                                f"or pairwise Mann-Whitney U tests with a Bonferroni correction for post-hoc comparisons.</div>"
                            )
                    except Exception:
                        pass
                elif self.anova_kw.isChecked():
                    out += ("<p style='color:#6B7280; font-size:13px;'><i>Kruskal-Wallis was requested but skipped: "
                            "it requires exactly one Dependent Variable, one Fixed Factor, and no Covariates.</i></p>")
                
                out += "<h2>Post-Hoc Pairwise Comparisons (Tukey HSD)</h2>"
                post_hoc_run = False
                for iv in ivs:
                    if len(data_sub[iv].unique()) > 2:
                        post_hoc_run = True
                        tukey = pairwise_tukeyhsd(endog=data_sub[dv], groups=data_sub[iv], alpha=0.05)
                        out += f"<h3>Differences in {dv} across {iv} groups</h3>"
                        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                        out += tukey_df.to_html(classes='apa', index=False)
                if not post_hoc_run:
                    out += "<p style='color:#6B7280;'><i>No post-hoc tests run because no categorical factor had more than 2 distinct groups.</i></p>"

                tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
                self.anova_tabs.addTab(tv, f"ANOVA ({timestamp})")

                # Raincloud plot is hard-coded to trigger alongside One-Way ANOVA
                # (single factor, no covariates, no interaction needed for a clean group comparison).
                if len(ivs) == 1 and not covs:
                    try:
                        group_dict = {str(k): v[dv].values for k, v in data_sub.groupby(first_iv)}
                        rc_widget = self.build_raincloud_plot(group_dict, dv, title=f"{dv} by {first_iv}")
                        if rc_widget is not None:
                            self.anova_tabs.addTab(rc_widget, f"Raincloud ({timestamp})")
                    except Exception:
                        pass
            
            else:
                dv_str = " + ".join(dvs)
                formula = f"{dv_str} ~ {factor_term}{cov_term}"
                manova = MANOVA.from_formula(formula, data=data_sub)
                res = manova.mv_test()
                
                out += "<h2>MANOVA Results (Multivariate Tests)</h2>"
                out += f"<p><b>Dependent Variables Analyzed:</b> {', '.join(dvs)}</p>"
                out += "<table class='apa'><tr><th>Effect</th><th>Statistic</th><th>Value</th><th>F-value</th><th>Num DF</th><th>Den DF</th><th>p-value</th></tr>"
                for effect in res.results:
                    if effect == 'Intercept': continue
                    stats_df = res.results[effect]['stat']
                    for stat_name in ['Wilks\' lambda', 'Pillai\'s trace']:
                        if stat_name in stats_df.index:
                            row = stats_df.loc[stat_name]
                            val = self.fmt(row['Value'])
                            f_v = self.fmt(row['F Value'])
                            ndf = self.fmt(row['Num DF'])
                            ddf = self.fmt(row['Den DF'])
                            p_v = row['Pr > F']
                            
                            p_str = "" if pd.isna(p_v) else (f"<span class='sig'>{self.fmt(p_v, True)}</span>" if p_v<0.05 else self.fmt(p_v, True))
                            out += f"<tr><td style='text-align:left;'><b>{effect}</b></td><td style='text-align:left;'>{stat_name}</td><td>{val}</td><td>{f_v}</td><td>{ndf}</td><td>{ddf}</td><td>{p_str}</td></tr>"
                
                out += "</table><div class='interpret'><i>Note.</i> <b>Interpretation (Hair et al., 2010):</b> MANOVA tests whether mean differences among groups on a combination of DVs are likely to have occurred by chance. A significant Wilks' Lambda or Pillai's Trace (p < .05) indicates the groups differ significantly on the combined dependent variables.</div>"
                
                out += "<h2>Univariate Post-Hoc Pairwise Comparisons (Tukey HSD)</h2>"
                post_hoc_run = False
                for curr_dv in dvs:
                    for curr_iv in ivs:
                        if len(data_sub[curr_iv].unique()) > 2:
                            post_hoc_run = True
                            tukey = pairwise_tukeyhsd(endog=data_sub[curr_dv], groups=data_sub[curr_iv], alpha=0.05)
                            out += f"<h3>Dependent Variable: {curr_dv} across {curr_iv} groups</h3>"
                            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                            out += tukey_df.to_html(classes='apa', index=False)
                if not post_hoc_run:
                    out += "<p style='color:#6B7280;'><i>No post-hoc tests run because no categorical factor had more than 2 distinct groups.</i></p>"

                tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
                self.anova_tabs.addTab(tv, f"MANOVA ({timestamp})")

            self.anova_tabs.setCurrentIndex(self.anova_tabs.count() - 1)
        except Exception as e:
            self.anova_tabs.addTab(QTextEdit(str(e)), "Error")

    def run_repeated_measures(self):
        """
        Runs Repeated-Measures ANOVA (parametric) and/or the Friedman test
        (non-parametric) on 3+ repeated-measure columns (e.g. Pretest/Midtest/
        Posttest) belonging to the same subjects. The app stores data wide (one
        column per timepoint), but statsmodels' AnovaRM requires long format
        (subject id, condition, value), so the wide selection is melted first.
        """
        if self.df is None: return
        measures = [i.text() for i in self.rm_vars.selectedItems()]
        if len(measures) < 3:
            QMessageBox.warning(self, "Select Variables", "Please select at least 3 repeated-measure columns (use the Paired T-Test in Compare Means for exactly 2 time points).")
            return

        try:
            data = self.df[measures].dropna().reset_index(drop=True)
            timestamp = datetime.now().strftime("%H:%M:%S")
            out = self.get_apa_css() + f"<h2>Repeated Measures: {' → '.join(measures)}</h2>"
            out += self.build_missing_data_note(self.df, measures, len(data))
            out += self.build_sample_size_warning(len(data), "anova")

            if self.rm_show_desc.isChecked():
                out += self.build_repeated_measures_descriptives_table(data, measures)

            if len(data) < 3:
                raise ValueError("Not enough complete cases (need at least 3 subjects with no missing data across all selected measures).")

            # Long-format reshape: one row per (subject, condition) pair, as required
            # by statsmodels.stats.anova.AnovaRM (Hsu, 1994 framework).
            data_long = data.copy()
            data_long['_subject_id'] = data_long.index
            long_df = pd.melt(data_long, id_vars=['_subject_id'], value_vars=measures,
                               var_name='_condition', value_name='_value')

            run_type = self.rm_test_type.currentText()
            run_parametric = "ANOVA" in run_type or run_type == "Both"
            run_friedman = "Friedman" in run_type or run_type == "Both"

            if run_parametric:
                try:
                    if PINGOUIN_AVAILABLE:
                        # pingouin.rm_anova provides Mauchly's W, Greenhouse-Geisser ε,
                        # Huynh-Feldt ε, and corrected p-values — matching SPSS GLM output.
                        pg_result = pg.rm_anova(
                            data=long_df, dv='_value',
                            within='_condition', subject='_subject_id',
                            detailed=True, correction='auto'
                        )
                        row0 = pg_result.iloc[0]
                        f_val = float(row0['F'])
                        df1   = float(row0['ddof1'])
                        df2   = float(row0['ddof2'])
                        p_val = float(row0['p-unc'])

                        # Sphericity diagnostics
                        has_spher = 'p-spher' in pg_result.columns
                        p_spher   = float(row0['p-spher']) if has_spher else None
                        w_spher   = float(row0['W-spher']) if 'W-spher' in pg_result.columns else None
                        eps_gg    = float(row0['eps'])     if 'eps'     in pg_result.columns else None
                        p_gg      = float(row0['p-GG-corr']) if 'p-GG-corr' in pg_result.columns else None

                        out += "<h2>Repeated-Measures ANOVA</h2>"
                        out += "<table class='apa'><tr><th>Source</th><th>F-value</th><th>df (Num)</th><th>df (Den)</th><th>p-value (uncorrected)</th></tr>"
                        p_str = f"<span class='sig'>{self.apa_p(p_val)}</span>" if p_val < 0.05 else self.apa_p(p_val)
                        out += f"<tr><td style='text-align:left;'>{', '.join(measures)}</td><td>{self.fmt(f_val)}</td><td>{self.fmt(df1)}</td><td>{self.fmt(df2)}</td><td>{p_str}</td></tr>"
                        out += "</table>"

                        # Sphericity section
                        if p_spher is not None:
                            spher_violated = p_spher < 0.05
                            spher_color  = "#EF4444" if spher_violated else "#10B981"
                            spher_icon   = "⚠" if spher_violated else "✔"
                            spher_status = "VIOLATED" if spher_violated else "Assumed"
                            out += (
                                f"<div style='background:#F9FAFB; border-left:4px solid {spher_color}; "
                                f"padding:8px 12px; margin:10px 0; font-size:13px;'>"
                                f"<b>Mauchly's Test of Sphericity:</b> "
                                f"W = {self.fmt(w_spher)}, {self.apa_p(p_spher)} &mdash; "
                                f"<span style='color:{spher_color}; font-weight:bold;'>{spher_icon} Sphericity {spher_status}</span>"
                            )
                            if eps_gg is not None:
                                gg_p_str = self.apa_p(p_gg) if p_gg is not None else "N/A"
                                gg_p_fmt = f"<span class='sig'>{gg_p_str}</span>" if p_gg is not None and p_gg < 0.05 else gg_p_str
                                out += (
                                    f"<br>Greenhouse-Geisser &epsilon; = {self.fmt(eps_gg)}, "
                                    f"GG-corrected p = {gg_p_fmt}"
                                )
                            if spher_violated:
                                out += (
                                    "<br><i>Recommendation: report the GG-corrected p-value above. "
                                    "If &epsilon; &ge; .75, Huynh-Feldt correction is a less conservative alternative.</i>"
                                )
                            out += "</div>"

                        out += (
                            "<div class='interpret'><i>Note.</i> <b>Interpretation (Field, 2013):</b> If p &lt; .05, "
                            "at least one timepoint/condition differs from the others. Sphericity is tested via "
                            "Mauchly's W; when violated, use the Greenhouse-Geisser corrected p-value.</div>"
                        )
                    else:
                        # Fallback: statsmodels AnovaRM (no sphericity correction)
                        aovrm = AnovaRM(long_df, depvar='_value', subject='_subject_id', within=['_condition'])
                        res = aovrm.fit()
                        res_table = res.anova_table
                        f_val = res_table.loc['_condition', 'F Value']
                        df1   = res_table.loc['_condition', 'Num DF']
                        df2   = res_table.loc['_condition', 'Den DF']
                        p_val = res_table.loc['_condition', 'Pr > F']

                        out += "<h2>Repeated-Measures ANOVA</h2>"
                        out += "<table class='apa'><tr><th>Source</th><th>F-value</th><th>df (Num)</th><th>df (Den)</th><th>p-value</th></tr>"
                        p_str = f"<span class='sig'>{self.apa_p(p_val)}</span>" if p_val < 0.05 else self.apa_p(p_val)
                        out += f"<tr><td style='text-align:left;'>{', '.join(measures)}</td><td>{self.fmt(f_val)}</td><td>{self.fmt(df1)}</td><td>{self.fmt(df2)}</td><td>{p_str}</td></tr>"
                        out += "</table>"
                        out += (
                            "<div class='interpret'><i>Note.</i> <b>Interpretation (Field, 2013):</b> If p &lt; .05, at least "
                            "one timepoint/condition differs from the others. <b>Warning:</b> Install <code>pingouin</code> "
                            "to enable Mauchly's sphericity test and GG/HF corrections.</div>"
                        )

                    if not pd.isna(p_val) and p_val < 0.05:
                        out += "<h3>Post-Hoc Pairwise Comparisons (Paired T-Tests, Bonferroni-corrected)</h3>"
                        out += self.build_pairwise_corrected_table(data, measures, paired=True)
                except Exception as e:
                    out += f"<p class='warn'>Repeated-Measures ANOVA could not be computed: {str(e)}</p>"

            if run_friedman:
                try:
                    arrays = [data[m].values for m in measures]
                    fr_stat, fr_p = stats.friedmanchisquare(*arrays)
                    out += "<h2>Friedman Test (Non-Parametric)</h2>"
                    fr_p_str = f"<span class='sig'>{self.apa_p(fr_p)}</span>" if fr_p < 0.05 else self.apa_p(fr_p)
                    out += f"<p>χ²({len(measures)-1}, N = {len(data)}) = {self.fmt(fr_stat)}, {fr_p_str}</p>"
                    out += (
                        "<div class='interpret'><i>Note.</i> The Friedman test (Friedman, 1937) is the non-parametric "
                        "equivalent of repeated-measures ANOVA: it ranks each subject's scores across conditions "
                        "instead of using raw values, so it does not require normally distributed residuals or "
                        "sphericity. A significant result (p &lt; .05) indicates at least one condition differs; "
                        "consider Wilcoxon signed-rank pairwise comparisons with a Bonferroni correction as a post-hoc "
                        "follow-up.</div>"
                    )
                    if not pd.isna(fr_p) and fr_p < 0.05:
                        out += "<h3>Post-Hoc Pairwise Comparisons (Wilcoxon Signed-Rank, Bonferroni-corrected)</h3>"
                        out += self.build_pairwise_corrected_table(data, measures, paired=True, nonparametric=True)
                except Exception as e:
                    out += f"<p class='warn'>Friedman test could not be computed: {str(e)}</p>"

            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
            self.anova_tabs.addTab(tv, f"Repeated Measures ({timestamp})")

            # Raincloud-style visual: one "group" per timepoint, reusing the same
            # raincloud builder used for independent t-tests / one-way ANOVA.
            try:
                group_dict = {m: data[m].values for m in measures}
                rc_widget = self.build_raincloud_plot(group_dict, "Value", title="Repeated Measures by Timepoint/Condition")
                if rc_widget is not None:
                    self.anova_tabs.addTab(rc_widget, f"Raincloud ({timestamp})")
            except Exception:
                pass

            self.anova_tabs.setCurrentIndex(self.anova_tabs.count() - 1)
        except Exception as e:
            self.anova_tabs.addTab(QTextEdit(str(e)), "Error")

    # ==========================================
    # MODULE 6: REGRESSION
    # ==========================================
    def init_regression_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.add_info_box(cl, "<b>Regression:</b> Determine how strongly independent variables predict a dependent variable. Includes Standard (OLS), Logistic, Polynomial, and Penalized (Ridge/Lasso) Regression models.")


        mg = QGroupBox("Model Configuration")
        mg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        mgl = QFormLayout(mg)
        mgl.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        mgl.setVerticalSpacing(5)
        
        self.reg_type = QComboBox()
        self.reg_type.addItems([
            "Linear Regression (OLS - Hierarchical & Moderation)", 
            "Logistic Regression (Binary Y)", 
            "Multinomial Logistic Regression (Nominal Y, 3+ Categories)",
            "Ordinal Logistic Regression (Ordered Y, 3+ Categories)",
            "Polynomial Regression (Quadratic)", 
            "Ridge Regression (L2 Regularization)", 
            "Lasso Regression (L1 Regularization)"
        ])
        self.reg_type.setToolTip(
            "Linear: continuous Y.\n"
            "Logistic: binary Y (exactly 2 categories), e.g. pass/fail.\n"
            "Multinomial Logistic: nominal Y with 3+ unordered categories, e.g. preferred brand A/B/C.\n"
            "Ordinal Logistic: ordered Y with 3+ ranked categories, e.g. low/medium/high satisfaction.\n"
            "Polynomial: continuous Y with a curved (quadratic) relationship to predictors.\n"
            "Ridge/Lasso: continuous Y with many/collinear predictors, trading some bias for stability."
        )
        mgl.addRow("Regression Model Type:", self.reg_type)
        
        self.reg_dv = QComboBox(); self.enable_drag_drop_combo(self.reg_dv); mgl.addRow("Dependent Variable (Y):", self.reg_dv)
        cl.addWidget(mg)
        
        pg = QGroupBox("Predictors (Independent Variables)")
        pg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        pgl = QVBoxLayout(pg)
        pgl.setSpacing(5)
        
        self.reg_block1 = QListWidget(); self.reg_block1.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.reg_block1.setMinimumHeight(80)
        self.setup_list_selection(pgl, "Block 1 (Controls / All Predictors for Non-OLS):", self.reg_block1, "Block 1 Predictors")
        
        self.reg_block2 = QListWidget(); self.reg_block2.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.reg_block2.setMinimumHeight(80)
        self.setup_list_selection(pgl, "Block 2 (Main Predictors - Hierarchical OLS Only):", self.reg_block2, "Block 2 Predictors")
        
        self.reg_block3 = QListWidget(); self.reg_block3.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.reg_block3.setMinimumHeight(80)
        self.setup_list_selection(pgl, "Block 3 (Additional Predictors / Interactions):", self.reg_block3, "Block 3 Predictors")
        
        self.reg_remove_outliers = QCheckBox("Remove Outliers Before Running (Cook's D > 4/N)")
        self.reg_remove_outliers.setChecked(False)
        self.reg_remove_outliers.setToolTip(
            "When checked, cases flagged by Cook's Distance (D > 4/N) are excluded and the model "
            "re-run without them. The number removed is reported in the output. Inspect the "
            "Assumptions tab first to see which specific cases are flagged."
        )
        pgl.addWidget(self.reg_remove_outliers)

        btn = QPushButton("▶ Run Regression")
        btn.setStyleSheet("margin-top: 10px;")
        btn.clicked.connect(self.run_regression)
        pgl.addWidget(btn)
        
        cl.addWidget(pg)
        
        tab, self.reg_tabs = self.create_split_module(cw, bank=self.build_bank_panel())
        self.tabs.addWidget(tab)

    def run_regression(self):
        if self.df is None: return
        dv = self.reg_dv.currentText()
        
        b1_raw = [i.text() for i in self.reg_block1.selectedItems()]
        b2_raw = [i.text() for i in self.reg_block2.selectedItems()]
        b3_raw = [i.text() for i in self.reg_block3.selectedItems()]
        
        b1 = list(dict.fromkeys(b1_raw))
        b2 = [x for x in b2_raw if x not in b1]
        b3 = [x for x in b3_raw if x not in b1 + b2]
        
        all_ivs = list(dict.fromkeys(b1 + b2 + b3))
        
        reg_type = self.reg_type.currentText()
        if not dv or not all_ivs: return
        
        try:
            data = self.df[[dv] + all_ivs].dropna()
            Y = data[dv]

            # Pre-screening outlier removal (Cook's D > 4/N) when requested
            if "Linear" in reg_type and hasattr(self, 'reg_remove_outliers') and self.reg_remove_outliers.isChecked():
                try:
                    import statsmodels.api as _sm
                    _X_pre = _sm.add_constant(data[all_ivs])
                    _m_pre = _sm.OLS(data[dv], _X_pre).fit()
                    _cooks, _ = _m_pre.get_influence().cooks_distance
                    _threshold = 4 / max(len(data), 1)
                    _outlier_mask = _cooks > _threshold
                    _n_removed = int(_outlier_mask.sum())
                    if _n_removed > 0:
                        data = data[~_outlier_mask].reset_index(drop=True)
                        Y = data[dv]
                        out += (f"<div style='background:#FEF3C7; border-left:4px solid #D97706; padding:8px 12px; "
                                f"margin:8px 0; font-size:13px; color:#78350F;'>"
                                f"⚠ <b>Outlier removal:</b> {_n_removed} case(s) with Cook's D > {_threshold:.4f} "
                                f"were excluded before fitting. N analysed = {len(data)}.</div>")
                except Exception:
                    pass  # If outlier detection fails, proceed with full dataset
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            out = self.get_apa_css() + f"<h2>Model Summary ({reg_type.split('(')[0].strip()})</h2>"
            out += self.build_missing_data_note(self.df, [dv] + all_ivs, len(data))

            cases_per_predictor = len(data) / max(len(all_ivs), 1)
            if cases_per_predictor < 10:
                out += (
                    "<div style='background:#FEF2F2; border-left:4px solid #EF4444; padding:8px 12px; "
                    "margin:10px 0; font-size:13px; color:#7F1D1D;'>"
                    f"⚠ <b>Low cases-to-predictor ratio:</b> N = {len(data)} with {len(all_ivs)} predictor(s) "
                    f"gives only {cases_per_predictor:.1f} cases per predictor. Regression models are commonly "
                    "recommended to have at least 10–20 cases per predictor to avoid overfitting and unstable "
                    "coefficient estimates (Green, 1991). Results are still reported below, but treat coefficients "
                    "and R² with caution.</div>"
                )
            
            out += f"<div style='background-color:#E0E7FF; border-left: 4px solid #4F46E5; padding: 10px; margin-bottom: 15px; color:#1E3A8A;'>"
            out += f"<b>Target / Dependent Variable (Y):</b> {dv}</div>"

            if "Linear" in reg_type:
                models = []
                
                if len(b1) > 0:
                    X1 = sm.add_constant(data[b1])
                    models.append(('Model 1 (Block 1)', sm.OLS(Y, X1).fit(), data[b1]))
                if len(b2) > 0:
                    X2 = sm.add_constant(data[b1 + b2])
                    models.append(('Model 2 (Block 1+2)', sm.OLS(Y, X2).fit(), data[b1 + b2]))
                if len(b3) > 0:
                    X3 = sm.add_constant(data[b1 + b2 + b3])
                    models.append(('Model 3 (Block 1+2+3)', sm.OLS(Y, X3).fit(), data[b1 + b2 + b3]))
                    
                if not models:
                    X_all = sm.add_constant(data[all_ivs])
                    models.append(('Final Model', sm.OLS(Y, X_all).fit(), data[all_ivs]))
                    
                final_model = models[-1][1]
                X_final = sm.add_constant(models[-1][2])
                res = final_model.resid
                
                asm = self.get_apa_css() + "<h2>Classic Assumption Tests (Final Model)</h2><table class='apa'><tr><th>Assumption</th><th>Metric</th><th>Value</th><th>Cut-off Guidelines</th><th>Status</th></tr>"
                
                vif_details = ""
                max_vif = 1
                if X_final.shape[1] > 1:
                    vifs = [(X_final.columns[i], variance_inflation_factor(X_final.values, i)) for i in range(1, X_final.shape[1])]
                    max_vif = max([v[1] for v in vifs]) if vifs else 1
                    vif_details = "<br>".join([f"<span style='font-size:12px;'>{k}: {self.fmt(v)}</span>" for k,v in vifs])
                
                v_stat = "<span class='sig'>Pass</span>" if max_vif < 10 else "<span class='warn'>Violated</span>"
                asm += f"<tr><td style='text-align:left; vertical-align:top;'>Multicollinearity</td><td style='vertical-align:top;'>Max VIF</td><td style='vertical-align:top;'>Max: <b>{self.fmt(max_vif)}</b><br><br>{vif_details}</td><td style='vertical-align:top;'>&lt; 10 (Hair et al., 2010)</td><td style='vertical-align:top;'>{v_stat}</td></tr>"
                
                dw = durbin_watson(res)
                dw_stat = "<span class='sig'>Pass</span>" if 1.5 <= dw <= 2.5 else "<span class='warn'>Violated</span>"
                asm += f"<tr><td style='text-align:left;'>Auto-Correlation</td><td>Durbin-Watson</td><td>{self.fmt(dw)}</td><td>1.5 - 2.5 (Field, 2013)</td><td>{dw_stat}</td></tr>"
                
                bp_p = het_breuschpagan(res, final_model.model.exog)[1]
                bp_stat = "<span class='sig'>Pass</span>" if bp_p > 0.05 else "<span class='warn'>Violated</span>"
                asm += f"<tr><td style='text-align:left;'>Homoscedasticity</td><td>Breusch-Pagan p-val</td><td>{self.fmt(bp_p, True)}</td><td>&gt; .05 (Field, 2013)</td><td>{bp_stat}</td></tr>"

                sk, ku = stats.skew(res, bias=False), stats.kurtosis(res, bias=False)
                norm_stat = "<span class='sig'>Pass</span>" if abs(sk)<2 and abs(ku)<7 else "<span class='warn'>Violated</span>"
                asm += f"<tr><td style='text-align:left;'>Normality of Residuals</td><td>Skew / Kurtosis</td><td>Sk: {self.fmt(sk)}<br>Ku: {self.fmt(ku)}</td><td>|Sk| &lt; 2, |Ku| &lt; 7 (Kim, 2013)</td><td>{norm_stat}</td></tr>"

                shapiro_w, shapiro_p = (np.nan, np.nan)
                if 3 <= len(res) <= 5000:
                    try:
                        shapiro_w, shapiro_p = stats.shapiro(res)
                    except Exception:
                        shapiro_w, shapiro_p = (np.nan, np.nan)
                if not np.isnan(shapiro_p):
                    sw_stat = "<span class='sig'>Pass</span>" if shapiro_p > 0.05 else "<span class='warn'>Violated</span>"
                    asm += f"<tr><td style='text-align:left;'>Normality of Residuals</td><td>Shapiro-Wilk</td><td>W = {self.fmt(shapiro_w)}<br>p = {self.fmt(shapiro_p, True)}</td><td>p &gt; .05 (Shapiro &amp; Wilk, 1965)</td><td>{sw_stat}</td></tr>"
                else:
                    asm += f"<tr><td style='text-align:left;'>Normality of Residuals</td><td>Shapiro-Wilk</td><td colspan='3' style='color:#9CA3AF;'>N/A (requires 3–5000 residuals)</td></tr>"

                # --- Outliers & Influential Cases (Cook's Distance, studentized residuals, leverage) ---
                outlier_table_html = ""
                try:
                    influence = final_model.get_influence()
                    cooks_d, _ = influence.cooks_distance
                    student_resid = influence.resid_studentized_external
                    leverage = influence.hat_matrix_diag
                    n_obs = len(cooks_d)
                    cooks_threshold = 4 / n_obs if n_obs > 0 else np.nan

                    n_flagged_cooks = int(np.sum(cooks_d > cooks_threshold)) if n_obs > 0 else 0
                    n_flagged_resid = int(np.sum(np.abs(student_resid) > 3))
                    max_cooks = np.max(cooks_d) if n_obs > 0 else np.nan

                    outlier_stat = "<span class='sig'>Pass</span>" if (n_flagged_cooks == 0 and n_flagged_resid == 0) else "<span class='warn'>Review</span>"
                    asm += (f"<tr><td style='text-align:left; vertical-align:top;'>Outliers &amp; Influence</td>"
                            f"<td style='vertical-align:top;'>Cook's D / Studentized Resid.</td>"
                            f"<td style='vertical-align:top;'>Max Cook's D: <b>{self.fmt(max_cooks)}</b><br>"
                            f"{n_flagged_cooks} case(s) &gt; 4/N<br>{n_flagged_resid} case(s) with |resid| &gt; 3</td>"
                            f"<td style='vertical-align:top;'>Cook's D &lt; 4/N (Cook, 1977)<br>|Studentized resid| &lt; 3</td>"
                            f"<td style='vertical-align:top;'>{outlier_stat}</td></tr>")

                    flagged_idx = sorted(set(np.where(cooks_d > cooks_threshold)[0]) | set(np.where(np.abs(student_resid) > 3)[0]))
                    if flagged_idx:
                        row_index_labels = data.index[flagged_idx] if hasattr(data, 'index') else flagged_idx
                        rows_html = ""
                        for pos, orig_idx in zip(flagged_idx, row_index_labels):
                            rows_html += (f"<tr><td>{orig_idx}</td><td>{self.fmt(cooks_d[pos])}</td>"
                                          f"<td>{self.fmt(student_resid[pos])}</td><td>{self.fmt(leverage[pos])}</td></tr>")
                        outlier_table_html = (
                            "<h3 style='margin-top:18px;'>Flagged Influential Cases</h3>"
                            "<table class='apa'><tr><th>Row (Dataset Index)</th><th>Cook's Distance</th><th>Studentized Residual</th><th>Leverage (h)</th></tr>"
                            f"{rows_html}</table>"
                            "<p style='font-size:12.5px; color:#6B7280;'><i>These cases disproportionately influence the regression fit. "
                            "Inspect them for data entry errors before deciding whether to retain, transform, or exclude them — "
                            "exclusion should be justified and reported, never done purely to improve significance.</i></p>"
                        )
                except Exception:
                    asm += "<tr><td style='text-align:left;'>Outliers &amp; Influence</td><td colspan='4' style='color:#9CA3AF;'>Could not be computed for this model.</td></tr>"
                
                asm += "</table><div class='interpret'><i>Note.</i> <b>Assumption Guidelines:</b><br>"
                asm += "<b>Multicollinearity (VIF):</b> Values &lt; 10 indicate no severe multicollinearity (Hair et al., 2010).<br>"
                asm += "<b>Auto-Correlation (Durbin-Watson):</b> Values between 1.5 and 2.5 are generally acceptable (Field, 2013).<br>"
                asm += "<b>Homoscedasticity (Breusch-Pagan):</b> A p-value &gt; .05 indicates residuals have roughly equal variance.<br>"
                asm += "<b>Normality (Skew/Kurtosis):</b> Absolute Skewness &lt; 2 and Kurtosis &lt; 7 suggest normal distribution of residuals (Kim, 2013).<br>"
                asm += "<b>Normality (Shapiro-Wilk):</b> A p-value &gt; .05 indicates the residuals do not significantly deviate from a normal distribution. Shapiro-Wilk is a formal significance test and is more sensitive than skew/kurtosis with larger samples (n &gt; ~300), where it can flag trivial deviations as significant — always interpret it alongside the Normal P-P/Q-Q plot rather than in isolation.<br>"
                asm += "<b>Outliers &amp; Influence:</b> Cook's Distance flags cases that disproportionately pull the regression coefficients; a common rule of thumb is D &gt; 4/N (Cook, 1977). Studentized residuals beyond ±3 flag cases the model fits poorly. Neither test alone proves a case should be removed.</div>"
                asm += outlier_table_html
                
                t1 = QTextEdit(); t1.setReadOnly(True); t1.setHtml(asm)
                self.reg_tabs.addTab(t1, f"Assumptions ({timestamp})")

                out += "<table class='apa'><tr><th>Model</th><th>R²</th><th>Adj. R²</th><th>R² Change</th><th>F Change</th><th>Sig. F Change</th></tr>"
                
                prev_r2 = 0
                prev_df1 = 0
                prev_df2 = 0
                
                for i, (m_name, m_fit, m_data) in enumerate(models):
                    r2 = m_fit.rsquared
                    df1 = m_fit.df_model
                    df2 = m_fit.df_resid
                    
                    if i == 0:
                        out += f"<tr><td style='text-align:left;'>{m_name}</td><td>{self.fmt(r2)}</td><td>{self.fmt(m_fit.rsquared_adj)}</td><td>{self.fmt(r2)}</td><td>{self.fmt(m_fit.fvalue)}</td><td>{self.fmt(m_fit.f_pvalue, True)}</td></tr>"
                    else:
                        r2_diff = r2 - prev_r2
                        f_change = (r2_diff / (df1 - prev_df1)) / ((1 - r2) / df2) if (df1 - prev_df1) > 0 else 0
                        p_change = stats.f.sf(f_change, df1 - prev_df1, df2) if (df1 - prev_df1) > 0 else 1.0
                        bold_p = f"<span class='sig'>{self.fmt(p_change, True)}</span>" if p_change < 0.05 else self.fmt(p_change, True)
                        
                        out += f"<tr><td style='text-align:left;'>{m_name}</td><td>{self.fmt(r2)}</td><td>{self.fmt(m_fit.rsquared_adj)}</td><td>{self.fmt(r2_diff)}</td><td>{self.fmt(f_change)}</td><td>{bold_p}</td></tr>"
                        
                    prev_r2 = r2
                    prev_df1 = df1
                    prev_df2 = df2
                    
                out += "</table>"
                
                for m_name, m_fit, m_data in models:
                    out += f"<h2>Coefficients ({m_name})</h2><table class='apa'><tr><th>Predictor</th><th>Unstandardized B</th><th>Std. Beta (β)</th><th>Std.Error</th><th>t-value</th><th>Sig. (p)</th></tr>"
                    for col in m_fit.params.index:
                        b = m_fit.params[col]
                        se = m_fit.bse[col]
                        t_val = m_fit.tvalues[col]
                        p = m_fit.pvalues[col]
                        
                        if col != 'const': std_b = b * (data[col].std() / Y.std())
                        else: std_b = "-"
                        
                        bp = f"<span class='sig'>{self.fmt(p, True)}</span>" if p < 0.05 else self.fmt(p, True)
                        out += f"<tr><td style='text-align:left;'>{col}</td><td>{self.fmt(b)}</td><td>{self.fmt(std_b) if std_b != '-' else '-'}</td><td>{self.fmt(se)}</td><td>{self.fmt(t_val)}</td><td>{bp}</td></tr>"
                    out += "</table>"
                    
                out += "<div class='interpret'><i>Note.</i> <b>Interpretation (Field, 2013):</b> Unstandardized B represents the unit change in Y for every 1 unit change in X. Standardized Beta (β) allows comparing the relative strength of predictors.</div>"
                
                out += self.build_apa_writeup("Regression", {
                    "dv": dv, "r2": final_model.rsquared, "f_val": final_model.fvalue,
                    "df1": int(final_model.df_model), "df2": int(final_model.df_resid),
                    "p": final_model.f_pvalue,
                })
                
                t_main = QTextEdit(); t_main.setReadOnly(True); t_main.setHtml(out)
                self.reg_tabs.addTab(t_main, f"Results ({timestamp})")
                
                if MATPLOTLIB_AVAILABLE:
                    fig = Figure(figsize=(7, 5))
                    ax = fig.add_subplot(111)
                    if self.is_dark_mode:
                        fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#374151')
                        ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                        ax.title.set_color('white'); ax.tick_params(colors='white')
                    
                    predicted = final_model.predict(X_final)
                    
                    # Standardizing for a cleaner journal-style plot
                    pred_std = (predicted - predicted.mean()) / predicted.std()
                    Y_std = (Y - Y.mean()) / Y.std()
                    
                    ax.scatter(pred_std, Y_std, alpha=0.6, color='#10B981', edgecolor='white', linewidth=0.5)
                    self.plot_smooth_fit(ax, pd.Series(pred_std), Y_std, '#EF4444')
                        
                    ax.set_xlabel("Predicted Values (Standardized)")
                    ax.set_ylabel("Observed Values (Standardized)")
                    ax.set_title("Observed vs. Predicted Regression Fit")
                    fig.tight_layout()
                    chart = self.make_zoomable_chart(fig)
                    self.reg_tabs.addTab(chart, f"Fit Plot ({timestamp})")

                    # --- SPSS-style Normal P-P Plot + Q-Q Plot of residuals ---
                    pp_widget = self.build_residual_normality_plots(res)
                    if pp_widget is not None:
                        self.reg_tabs.addTab(pp_widget, f"Normal P-P / Q-Q ({timestamp})")
            
            elif "Multinomial" in reg_type:
                if Y.nunique() < 3:
                    raise ValueError("Multinomial Logistic Regression requires the Dependent Variable to have 3 or more categories. For exactly 2 categories, use 'Logistic Regression (Binary Y)' instead.")

                y_cat = Y.astype('category')
                categories = list(y_cat.cat.categories)
                y_codes = y_cat.cat.codes  # 0-indexed integer codes; statsmodels treats the lowest code as the reference category
                ref_category = categories[0]

                X2 = sm.add_constant(data[all_ivs])
                final_model = sm.MNLogit(y_codes, X2).fit(disp=0)

                _n_mn = len(y_codes)
                _ll0_mn  = final_model.llnull
                _llm_mn  = final_model.llf
                _cs_mn   = 1.0 - np.exp((2.0 / _n_mn) * (_ll0_mn - _llm_mn))
                _nk_mn   = _cs_mn / (1.0 - np.exp(2.0 * _ll0_mn / _n_mn))
                out += "<table class='apa'><tr><th>Metric</th><th>Value</th></tr>"
                out += f"<tr><td>Log-Likelihood (null model)</td><td>{self.fmt(_ll0_mn)}</td></tr>"
                out += f"<tr><td>Log-Likelihood (fitted model)</td><td>{self.fmt(_llm_mn)}</td></tr>"
                out += f"<tr><td>Pseudo R² — McFadden</td><td>{self.fmt(final_model.prsquared)}</td></tr>"
                out += f"<tr><td>Pseudo R² — Cox &amp; Snell <i>(SPSS default)</i></td><td>{self.fmt(_cs_mn)}</td></tr>"
                out += f"<tr><td>Pseudo R² — Nagelkerke <i>(SPSS default)</i></td><td>{self.fmt(_nk_mn)}</td></tr>"
                out += f"<tr><td>LLR p-value (omnibus model test)</td><td>{self.fmt(final_model.llr_pvalue, True)}</td></tr>"
                out += f"<tr><td>Reference Category</td><td>{ref_category}</td></tr>"
                out += "</table>"
                out += ("<div style='background:#EFF6FF; border-left:4px solid #3B82F6; padding:8px 12px; "
                        "margin:10px 0; font-size:12px; color:#1E3A8A;'>"
                        "<i>Note on R²:</i> SPSS reports Cox &amp; Snell and Nagelkerke by default. "
                        "McFadden values of 0.20–0.40 indicate excellent fit despite appearing lower. "
                        "Nagelkerke R² is rescaled to a 0–1 range and is most comparable to OLS R².</div>")
                out += (
                    f"<div class='interpret'><i>Note.</i> All coefficients below describe the log-odds of being in "
                    f"each category <b>relative to the reference category ({ref_category})</b> -- the category "
                    f"statsmodels assigns the lowest internal code to, which here is the first category in "
                    f"alphabetical/sorted order. There is one full set of coefficients per non-reference category.</div>"
                )

                # final_model.params is a DataFrame: rows = predictors, columns = one per
                # non-reference outcome category (statsmodels' MNLogit convention).
                for col_idx in final_model.params.columns:
                    cat_label = categories[col_idx + 1] if col_idx + 1 < len(categories) else f"Category {col_idx + 1}"
                    out += f"<h2>{cat_label} vs. {ref_category}</h2>"
                    out += "<table class='apa'><tr><th>Predictor</th><th>Log-Odds (B)</th><th>Odds Ratio (exp(B))</th><th>Std.Error</th><th>z-value</th><th>Sig. (p)</th></tr>"
                    for row in final_model.params.index:
                        b = final_model.params.loc[row, col_idx]
                        odds = np.exp(b)
                        se = final_model.bse.loc[row, col_idx]
                        z_val = final_model.tvalues.loc[row, col_idx]
                        p = final_model.pvalues.loc[row, col_idx]
                        bp = f"<span class='sig'>{self.fmt(p, True)}</span>" if p < 0.05 else self.fmt(p, True)
                        out += f"<tr><td style='text-align:left;'>{row}</td><td>{self.fmt(b)}</td><td>{self.fmt(odds)}</td><td>{self.fmt(se)}</td><td>{self.fmt(z_val)}</td><td>{bp}</td></tr>"
                    out += "</table>"

            elif "Ordinal" in reg_type:
                if Y.nunique() < 3:
                    raise ValueError("Ordinal Logistic Regression requires the Dependent Variable to have 3 or more ordered categories. For exactly 2 categories, use 'Logistic Regression (Binary Y)' instead.")

                y_cat = Y.astype('category')
                categories = list(y_cat.cat.categories)

                # OrderedModel's parameterization requires NO constant (explicit or implicit) --
                # adding one would make the thresholds unidentifiable (statsmodels docs). This is
                # the one regression branch in this app that must NOT call sm.add_constant().
                X_ord = data[all_ivs]
                final_model = OrderedModel(y_cat, X_ord, distr='logit').fit(method='bfgs', disp=0)

                n_thresholds = len(categories) - 1
                out += "<table class='apa'><tr><th>Metric</th><th>Value</th></tr>"
                out += f"<tr><td>Log-Likelihood</td><td>{self.fmt(final_model.llf)}</td></tr>"
                out += f"<tr><td>Category Order (low → high)</td><td>{' &lt; '.join(str(c) for c in categories)}</td></tr>"
                out += "</table>"
                out += (
                    f"<div style='background:#FFFBEB; border-left:4px solid #D97706; padding:8px 12px; margin:10px 0; "
                    f"font-size:13px; color:#78350F;'>⚠ <b>Category order matters:</b> this model assumes the "
                    f"category order shown above (alphabetical/sorted by default) reflects the TRUE low-to-high "
                    f"ordering of your outcome. If your categories are coded as text (e.g. \"Low\", \"Medium\", "
                    f"\"High\"), verify this sorts correctly -- if not, recode the variable numerically first "
                    f"(Data Management → Recode) so 1 = lowest, 2 = next, etc.</div>"
                )

                out += "<h2>Coefficients (Proportional Odds)</h2><table class='apa'><tr><th>Predictor</th><th>Log-Odds (B)</th><th>Odds Ratio (exp(B))</th><th>Std.Error</th><th>z-value</th><th>Sig. (p)</th></tr>"
                for row in all_ivs:
                    b = final_model.params[row]
                    odds = np.exp(b)
                    se = final_model.bse[row]
                    z_val = final_model.tvalues[row]
                    p = final_model.pvalues[row]
                    bp = f"<span class='sig'>{self.fmt(p, True)}</span>" if p < 0.05 else self.fmt(p, True)
                    out += f"<tr><td style='text-align:left;'>{row}</td><td>{self.fmt(b)}</td><td>{self.fmt(odds)}</td><td>{self.fmt(se)}</td><td>{self.fmt(z_val)}</td><td>{bp}</td></tr>"
                out += "</table>"

                out += "<h3>Threshold (Cutpoint) Parameters</h3><table class='apa'><tr><th>Threshold</th><th>Value</th></tr>"
                threshold_params = final_model.params.index[len(all_ivs):]
                for i, t_name in enumerate(threshold_params):
                    if i < n_thresholds:
                        out += f"<tr><td style='text-align:left;'>{categories[i]} | {categories[i+1]}</td><td>{self.fmt(final_model.params[t_name])}</td></tr>"
                out += "</table>"
                out += (
                    "<div class='interpret'><i>Note.</i> <b>Interpretation:</b> This is a proportional-odds model: "
                    "each predictor's effect on the odds of being in a higher vs. lower category is assumed constant "
                    "across all category thresholds. The Odds Ratio (exp(B)) represents the change in odds of being "
                    "in a higher category for a 1-unit increase in the predictor. This implementation does not test "
                    "the proportional-odds (parallel lines) assumption itself -- if predictors plausibly affect "
                    "different thresholds differently, treat results with caution.</div>"
                )

            elif "Logistic" in reg_type:
                if Y.nunique() != 2:
                    raise ValueError("Logistic regression requires the Dependent Variable to be Binary (exactly 2 unique values).")
                
                y_mapped = (Y == Y.unique()[1]).astype(int)
                X2 = sm.add_constant(data[all_ivs])
                final_model = sm.Logit(y_mapped, X2).fit(disp=0)
                
                _n_lg  = len(y_mapped)
                _ll0_lg = final_model.llnull
                _llm_lg = final_model.llf
                _cs_lg  = 1.0 - np.exp((2.0 / _n_lg) * (_ll0_lg - _llm_lg))
                _nk_lg  = _cs_lg / (1.0 - np.exp(2.0 * _ll0_lg / _n_lg))
                out += "<table class='apa'><tr><th>Metric</th><th>Value</th></tr>"
                out += f"<tr><td>Log-Likelihood (null model)</td><td>{self.fmt(_ll0_lg)}</td></tr>"
                out += f"<tr><td>Log-Likelihood (fitted model)</td><td>{self.fmt(_llm_lg)}</td></tr>"
                out += f"<tr><td>Pseudo R² — McFadden</td><td>{self.fmt(final_model.prsquared)}</td></tr>"
                out += f"<tr><td>Pseudo R² — Cox &amp; Snell <i>(SPSS default)</i></td><td>{self.fmt(_cs_lg)}</td></tr>"
                out += f"<tr><td>Pseudo R² — Nagelkerke <i>(SPSS default)</i></td><td>{self.fmt(_nk_lg)}</td></tr>"
                out += f"<tr><td>LLR p-value</td><td>{self.fmt(final_model.llr_pvalue, True)}</td></tr>"
                out += "</table>"
                out += ("<div style='background:#EFF6FF; border-left:4px solid #3B82F6; padding:8px 12px; "
                        "margin:10px 0; font-size:12px; color:#1E3A8A;'>"
                        "<i>Note on R²:</i> SPSS reports Cox &amp; Snell and Nagelkerke by default. "
                        "McFadden values of 0.20–0.40 indicate excellent fit despite appearing lower. "
                        "Nagelkerke R² is rescaled to a 0–1 range and is most comparable to OLS R².</div>")
                
                out += "<h2>Coefficients</h2><table class='apa'><tr><th>Predictor</th><th>Log-Odds (B)</th><th>Odds Ratio (exp(B))</th><th>Std.Error</th><th>z-value</th><th>Sig. (p)</th></tr>"
                for col in final_model.params.index:
                    b = final_model.params[col]
                    odds = np.exp(b)
                    se = final_model.bse[col]
                    z_val = final_model.tvalues[col]
                    p = final_model.pvalues[col]
                    
                    bp = f"<span class='sig'>{self.fmt(p, True)}</span>" if p < 0.05 else self.fmt(p, True)
                    out += f"<tr><td style='text-align:left;'>{col}</td><td>{self.fmt(b)}</td><td>{self.fmt(odds)}</td><td>{self.fmt(se)}</td><td>{self.fmt(z_val)}</td><td>{bp}</td></tr>"
                out += "</table><div class='interpret'><i>Note.</i> <b>Interpretation:</b> The Odds Ratio (exp(B)) represents the change in odds of Y occurring for a 1-unit increase in X. Values > 1 indicate increased likelihood.</div>"

            elif "Polynomial" in reg_type:
                X_poly = data[all_ivs].copy()
                for col in all_ivs:
                    X_poly[f"{col}^2"] = X_poly[col] ** 2
                
                X2 = sm.add_constant(X_poly)
                final_model = sm.OLS(Y, X2).fit()
                
                out += "<table class='apa'><tr><th>Model</th><th>R²</th><th>Adj. R²</th><th>F-value</th><th>Sig. F Change</th></tr>"
                out += f"<tr><td>Polynomial Model (Degree 2)</td><td>{self.fmt(final_model.rsquared)}</td><td>{self.fmt(final_model.rsquared_adj)}</td><td>{self.fmt(final_model.fvalue)}</td><td>{self.fmt(final_model.f_pvalue, True)}</td></tr>"
                out += "</table>"
                
                out += "<h2>Coefficients</h2><table class='apa'><tr><th>Predictor</th><th>Unstandardized B</th><th>Std.Error</th><th>t-value</th><th>Sig. (p)</th></tr>"
                for col in final_model.params.index:
                    b, se, t_val, p = final_model.params[col], final_model.bse[col], final_model.tvalues[col], final_model.pvalues[col]
                    bp = f"<span class='sig'>{self.fmt(p, True)}</span>" if p < 0.05 else self.fmt(p, True)
                    out += f"<tr><td style='text-align:left;'>{col}</td><td>{self.fmt(b)}</td><td>{self.fmt(se)}</td><td>{self.fmt(t_val)}</td><td>{bp}</td></tr>"
                out += "</table>"
                out += self.build_apa_writeup("Regression", {
                    "dv": dv, "r2": final_model.rsquared, "f_val": final_model.fvalue,
                    "df1": int(final_model.df_model), "df2": int(final_model.df_resid),
                    "p": final_model.f_pvalue,
                })

            elif "Ridge" in reg_type or "Lasso" in reg_type:
                if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is required for Ridge/Lasso.")
                
                X2 = data[all_ivs]
                X2_std = (X2 - X2.mean()) / X2.std()
                
                if "Ridge" in reg_type: model = Ridge(alpha=1.0)
                else: model = Lasso(alpha=0.1)
                
                model.fit(X2_std, Y)
                r2 = model.score(X2_std, Y)
                
                out += "<table class='apa'><tr><th>Metric</th><th>Value</th></tr>"
                out += f"<tr><td>R-squared</td><td>{self.fmt(r2)}</td></tr>"
                out += f"<tr><td>Regularization Parameter (Alpha)</td><td>{model.alpha}</td></tr>"
                out += "</table>"
                
                out += "<h2>Penalized Coefficients (Standardized Predictors)</h2><table class='apa'><tr><th>Predictor</th><th>Coefficient</th></tr>"
                out += f"<tr><td style='text-align:left;'>Constant (Intercept)</td><td>{self.fmt(model.intercept_)}</td></tr>"
                for i, col in enumerate(all_ivs):
                    out += f"<tr><td style='text-align:left;'>{col}</td><td>{self.fmt(model.coef_[i])}</td></tr>"
                out += "</table><div class='interpret'><i>Note.</i> <b>Note:</b> Penalized regressions (Ridge/Lasso) shrink coefficients to reduce overfitting and handle high multicollinearity. They do <b>not</b> produce standard p-values or standard errors. Variables forced to 0 (in Lasso) are entirely excluded from the model.</div>"

            if "Linear" not in reg_type:
                t_main = QTextEdit(); t_main.setReadOnly(True); t_main.setHtml(out)
                self.reg_tabs.addTab(t_main, f"Results ({timestamp})")
                
            self.reg_tabs.setCurrentIndex(self.reg_tabs.count() - 1)
        except Exception as e:
            self.reg_tabs.addTab(QTextEdit(str(e)), "Error")


    # ==========================================
    # MODULE 6.4: MEDIATION ANALYSIS
    # ==========================================
    def init_mediation_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.add_info_box(cl, "<b>Mediation Analysis:</b> Tests whether the effect of an Independent Variable (X) on a Dependent Variable (Y) operates indirectly through a Mediator (M). Reports paths a (X→M), b (M→Y), c (total effect), and c′ (direct effect), plus the indirect effect (a×b) with bootstrap and Sobel significance tests.")


        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.med_x = QComboBox()
        self.med_m = QComboBox()
        self.med_y = QComboBox()
        self.enable_drag_drop_combo(self.med_x)
        self.enable_drag_drop_combo(self.med_m)
        self.enable_drag_drop_combo(self.med_y)
        form.addRow(QLabel("Independent Variable (X):"), self.med_x)
        form.addRow(QLabel("Mediator (M):"), self.med_m)
        form.addRow(QLabel("Dependent Variable (Y):"), self.med_y)
        cl.addLayout(form)

        opt_g = QGroupBox("Indirect Effect Significance Test")
        opt_l = QFormLayout(opt_g)
        opt_l.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.med_n_boot = QSpinBox()
        self.med_n_boot.setRange(500, 10000)
        self.med_n_boot.setSingleStep(500)
        self.med_n_boot.setValue(2000)
        opt_l.addRow("Bootstrap Resamples:", self.med_n_boot)
        self.med_ci_level = QComboBox()
        self.med_ci_level.addItems(["95% CI", "90% CI", "99% CI"])
        opt_l.addRow("Confidence Level:", self.med_ci_level)
        cl.addWidget(opt_g)

        btn = QPushButton("▶ Run Mediation Analysis")
        btn.setStyleSheet("margin-top: 10px;")
        btn.clicked.connect(self.run_mediation)
        cl.addWidget(btn)
        cl.addStretch()

        tab, self.med_tabs = self.create_split_module(cw, bank=self.build_bank_panel(numeric_only=True))
        self.tabs.addWidget(tab)

    def _mediation_fit_paths(self, data, x, m, y):
        """
        Fits the three OLS regressions needed for simple mediation (Baron & Kenny, 1986 /
        Hayes, 2017 Model 4): a (X->M), b & c' (Y ~ X + M), and c (Y ~ X, total effect).
        Returns a dict of coefficients, SEs, and p-values for each path.
        """
        Xc = sm.add_constant(data[[x]])
        model_a = sm.OLS(data[m], Xc).fit()

        XMc = sm.add_constant(data[[x, m]])
        model_bc = sm.OLS(data[y], XMc).fit()

        model_c = sm.OLS(data[y], Xc).fit()

        return {
            'a': model_a.params[x], 'a_se': model_a.bse[x], 'a_p': model_a.pvalues[x],
            'b': model_bc.params[m], 'b_se': model_bc.bse[m], 'b_p': model_bc.pvalues[m],
            'cprime': model_bc.params[x], 'cprime_se': model_bc.bse[x], 'cprime_p': model_bc.pvalues[x],
            'c': model_c.params[x], 'c_se': model_c.bse[x], 'c_p': model_c.pvalues[x],
            'model_a': model_a, 'model_bc': model_bc, 'model_c': model_c,
        }

    def run_mediation(self):
        if self.df is None: return
        x, m, y = self.med_x.currentText(), self.med_m.currentText(), self.med_y.currentText()
        if not x or not m or not y:
            QMessageBox.warning(self, "Missing Variables", "Please select X, M, and Y variables.")
            return
        if len({x, m, y}) < 3:
            QMessageBox.warning(self, "Invalid Selection", "X, M, and Y must be three different variables.")
            return

        try:
            data = self.df[[x, m, y]].dropna()
            if len(data) < 10:
                raise ValueError("Not enough complete observations (need at least 10) for X, M, and Y combined.")

            res = self._mediation_fit_paths(data, x, m, y)
            a, b, cprime, c = res['a'], res['b'], res['cprime'], res['c']
            indirect = a * b

            # --- Bootstrap percentile CI (primary) ---
            ci_pct = {"95% CI": 95, "90% CI": 90, "99% CI": 99}.get(self.med_ci_level.currentText(), 95)
            alpha_tail = (100 - ci_pct) / 2
            n_boot = self.med_n_boot.value()
            rng = np.random.default_rng(42)
            n_rows = len(data)
            boot_indirect = np.empty(n_boot)
            boot_indirect[:] = np.nan
            for i in range(n_boot):
                idx = rng.integers(0, n_rows, n_rows)
                boot_df = data.iloc[idx].reset_index(drop=True)
                try:
                    r = self._mediation_fit_paths(boot_df, x, m, y)
                    boot_indirect[i] = r['a'] * r['b']
                except Exception:
                    continue
            boot_indirect = boot_indirect[~np.isnan(boot_indirect)]
            if len(boot_indirect) > 50:
                ci_lo, ci_hi = np.percentile(boot_indirect, [alpha_tail, 100 - alpha_tail])
                boot_n_used = len(boot_indirect)
            else:
                ci_lo, ci_hi = np.nan, np.nan
                boot_n_used = len(boot_indirect)

            # --- Sobel test (secondary) ---
            sobel_se = np.sqrt(b**2 * res['a_se']**2 + a**2 * res['b_se']**2)
            sobel_z = indirect / sobel_se if sobel_se > 0 else np.nan
            sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z))) if not np.isnan(sobel_z) else np.nan

            timestamp = datetime.now().strftime("%H:%M:%S")
            out = self.get_apa_css() + f"<h2>Mediation Analysis: {x} → {m} → {y}</h2>"
            out += self.build_missing_data_note(self.df, [x, m, y], len(data))
            out += f"<p><b>N (complete cases):</b> {len(data)} &nbsp;|&nbsp; <b>Bootstrap resamples used:</b> {boot_n_used}/{n_boot}</p>"
            out += self.build_sample_size_warning(len(data), "mediation")

            out += "<h2>Path Coefficients</h2>"
            out += "<table class='apa'><tr><th>Path</th><th>Description</th><th>Coefficient (B)</th><th>SE</th><th>p-value</th></tr>"
            out += f"<tr><td><b>a</b></td><td style='text-align:left;'>{x} → {m}</td><td>{self.fmt(a)}</td><td>{self.fmt(res['a_se'])}</td><td>{self.apa_p(res['a_p'])}</td></tr>"
            out += f"<tr><td><b>b</b></td><td style='text-align:left;'>{m} → {y} (controlling for {x})</td><td>{self.fmt(b)}</td><td>{self.fmt(res['b_se'])}</td><td>{self.apa_p(res['b_p'])}</td></tr>"
            out += f"<tr><td><b>c</b></td><td style='text-align:left;'>{x} → {y} (total effect, no mediator)</td><td>{self.fmt(c)}</td><td>{self.fmt(res['c_se'])}</td><td>{self.apa_p(res['c_p'])}</td></tr>"
            out += f"<tr><td><b>c′</b></td><td style='text-align:left;'>{x} → {y} (direct effect, controlling for {m})</td><td>{self.fmt(cprime)}</td><td>{self.fmt(res['cprime_se'])}</td><td>{self.apa_p(res['cprime_p'])}</td></tr>"
            out += "</table>"

            sig_indirect = (not np.isnan(ci_lo)) and (ci_lo > 0 or ci_hi < 0)
            box_color = "#10B981" if sig_indirect else "#9CA3AF"
            out += "<div style='display:flex; gap:16px; margin:16px 0;'>"
            out += f"<div style='flex:1; background:#ECFDF5; border:2px solid {box_color}; border-radius:10px; padding:14px; text-align:center;'>"
            out += "<div style='font-size:12px; color:#047857; font-weight:bold;'>INDIRECT EFFECT (a × b)</div>"
            out += f"<div style='font-size:26px; font-weight:900; color:#047857;'>{self.fmt(indirect)}</div>"
            ci_str = f"[{self.fmt(ci_lo)}, {self.fmt(ci_hi)}]" if not np.isnan(ci_lo) else "N/A"
            out += f"<div style='font-size:13px; color:#047857;'>{ci_pct}% Bootstrap CI: {ci_str}</div></div>"
            out += "<div style='flex:1; background:#EEF2FF; border:2px solid #4F46E5; border-radius:10px; padding:14px; text-align:center;'>"
            out += "<div style='font-size:12px; color:#4338CA; font-weight:bold;'>SOBEL TEST (SECONDARY)</div>"
            out += f"<div style='font-size:26px; font-weight:900; color:#4338CA;'>z = {self.fmt(sobel_z)}</div>"
            out += f"<div style='font-size:13px; color:#4338CA;'>{self.apa_p(sobel_p)}</div></div>"
            out += "</div>"

            if sig_indirect:
                concl = "The bootstrap confidence interval for the indirect effect <b>excludes zero</b>, indicating a statistically significant indirect (mediated) effect."
            else:
                concl = "The bootstrap confidence interval for the indirect effect <b>includes zero</b>, indicating the indirect (mediated) effect is not statistically significant at this confidence level."
            out += f"<div class='interpret'><i>Note.</i> {concl} <b>Bootstrapping (Preacher & Hayes, 2004, 2008)</b> is recommended over the Sobel test because it does not assume the indirect effect is normally distributed (its sampling distribution is typically skewed). The Sobel test is reported as a supplementary, more conservative check.</div>"

            apa_d = {
                "x": x, "m": m, "y": y, "a": a, "a_p": res['a_p'], "b": b, "b_p": res['b_p'],
                "c": c, "c_p": res['c_p'], "cprime": cprime, "cprime_p": res['cprime_p'],
                "indirect": indirect, "ci_lo": ci_lo, "ci_hi": ci_hi,
            }
            out += self.build_apa_writeup("Mediation", apa_d)

            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
            self.med_tabs.addTab(tv, f"Mediation Results ({timestamp})")
            self.med_tabs.setCurrentIndex(self.med_tabs.count() - 1)

            # --- Path diagram ---
            if MATPLOTLIB_AVAILABLE:
                try:
                    diagram_widget = self.build_mediation_path_diagram(x, m, y, res, indirect)
                    if diagram_widget is not None:
                        self.med_tabs.addTab(diagram_widget, f"Path Diagram ({timestamp})")
                except Exception:
                    pass

            # --- Bootstrap distribution histogram ---
            if MATPLOTLIB_AVAILABLE and len(boot_indirect) > 50:
                try:
                    hist_widget = self.build_mediation_bootstrap_hist(boot_indirect, ci_lo, ci_hi, indirect)
                    if hist_widget is not None:
                        self.med_tabs.addTab(hist_widget, f"Bootstrap Distribution ({timestamp})")
                except Exception:
                    pass

            self.med_tabs.setCurrentIndex(self.med_tabs.count() - 1)
        except Exception as e:
            self.med_tabs.addTab(QTextEdit(str(e)), "Error")

    def build_mediation_path_diagram(self, x_name, m_name, y_name, res, indirect):
        """
        Renders a publication-ready triangular mediation path diagram (X bottom-left,
        Y bottom-right, M top-center) with the a, b, and c' coefficients printed
        directly on their arrows, as is standard in mediation figures (Hayes, 2017).
        """
        from matplotlib.patches import FancyArrowPatch

        a, a_p = res['a'], res['a_p']
        b, b_p = res['b'], res['b_p']
        cprime, cprime_p = res['cprime'], res['cprime_p']
        is_dark = self.is_dark_mode

        def sig_stars(p):
            if pd.isna(p): return ""
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            return ""

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        if is_dark:
            fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#1F2937')

        pos = {'X': np.array([0.0, 0.0]), 'M': np.array([0.5, 0.9]), 'Y': np.array([1.0, 0.0])}
        labels = {'X': x_name, 'M': m_name, 'Y': y_name}

        box_fc = '#2A3A5C' if is_dark else '#EEF2FF'
        text_c = 'white' if is_dark else '#1E1B4B'
        box_style = dict(boxstyle="round,pad=0.5", fc=box_fc, ec='#4F46E5', lw=2)
        for node, (px, py) in pos.items():
            ax.text(px, py, labels[node], ha='center', va='center', fontsize=12, fontweight='bold',
                    color=text_c, bbox=box_style, zorder=5)

        def draw_path(p1, p2, label, weight, p_val, color, shrink=42, label_offset=(0, 0)):
            arrow = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=22, lw=2.2,
                                     color=color, shrinkA=shrink, shrinkB=shrink, zorder=2)
            ax.add_patch(arrow)
            mid = ((p1[0] + p2[0]) / 2 + label_offset[0], (p1[1] + p2[1]) / 2 + label_offset[1])
            ax.text(mid[0], mid[1], f"{label} = {self.fmt(weight)}{sig_stars(p_val)}",
                    ha='center', va='center', fontsize=10.5, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', fc=('#111827' if is_dark else 'white'),
                              ec=color, alpha=0.95, lw=0.8), zorder=6)

        draw_path(pos['X'], pos['M'], 'a', a, a_p, color='#818CF8' if is_dark else '#4F46E5', label_offset=(-0.07, 0))
        draw_path(pos['M'], pos['Y'], 'b', b, b_p, color='#818CF8' if is_dark else '#4F46E5', label_offset=(0.07, 0))
        draw_path(pos['X'], pos['Y'], "c'", cprime, cprime_p, color='#10B981', shrink=48, label_offset=(0, 0.07))

        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(-0.25, 1.2)
        ax.axis('off')
        ax.set_title(f"Mediation Path Diagram\nIndirect effect (a × b) = {self.fmt(indirect)}",
                     fontweight='bold', fontsize=13, color=text_c)
        fig.tight_layout()

        chart = self.make_zoomable_chart(fig)
        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)
        note = QLabel("* p < .05, ** p < .01, *** p < .001. c' is the direct effect of X on Y controlling for M.")
        note.setObjectName("InfoLabel")
        note.setWordWrap(True)
        wl.addWidget(note)
        wl.addWidget(chart)
        return wrapper

    def build_mediation_bootstrap_hist(self, boot_indirect, ci_lo, ci_hi, point_estimate):
        """Histogram of the bootstrap sampling distribution of the indirect effect (a*b)."""
        is_dark = self.is_dark_mode
        fig = Figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        if is_dark:
            fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#1F2937')
        text_c = 'white' if is_dark else 'black'

        ax.hist(boot_indirect, bins=40, color='#4F46E5', alpha=0.75, edgecolor='none')
        ax.axvline(point_estimate, color='#059669', linewidth=2.2, label=f'Point estimate = {self.fmt(point_estimate)}')
        if not np.isnan(ci_lo):
            ax.axvline(ci_lo, color='#EF4444', linewidth=1.6, linestyle='--', label=f'CI bounds [{self.fmt(ci_lo)}, {self.fmt(ci_hi)}]')
            ax.axvline(ci_hi, color='#EF4444', linewidth=1.6, linestyle='--')
        ax.axvline(0, color=text_c, linewidth=1.0, linestyle=':')
        ax.set_xlabel("Bootstrapped Indirect Effect (a × b)", color=text_c)
        ax.set_ylabel("Frequency", color=text_c)
        ax.tick_params(colors=text_c)
        ax.set_title("Bootstrap Sampling Distribution of the Indirect Effect", fontweight='bold', color=text_c, fontsize=12)
        legend = ax.legend(facecolor='#374151' if is_dark else 'white', labelcolor=text_c, fontsize=9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

        chart = self.make_zoomable_chart(fig)
        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)
        wl.addWidget(chart)
        return wrapper


    # ==========================================
    # MODULE 6.1: CATPCA (Categorical PCA)
    # ==========================================
    def init_catpca_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        if not SKLEARN_AVAILABLE: 
            lbl = QLabel("Missing module: scikit-learn. Cannot run CATPCA.")
            lbl.setWordWrap(True); lbl.setMinimumWidth(10)
            cl.addWidget(lbl)
            self.tabs.addWidget(cw); return
            
        self.add_info_box(cl, "<b>Categorical Principal Component Analysis (CATPCA):</b> A non-linear version of PCA suited for variables that are not strictly numeric (ordinal or nominal data). The analysis dynamically utilizes Optimal Scaling through Ordinal Encoding to calculate dimensional space. Use this to reduce dimensions of survey items or mixed-type data.")

        cg = QGroupBox("Configuration")
        cg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        cgl = QVBoxLayout(cg)
        cgl.setSpacing(5)

        self.catpca_vars = QListWidget(); self.catpca_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.catpca_vars.setMinimumHeight(150)
        self.setup_list_selection(cgl, "Select Variables to Reduce (Mixed Data allowed):", self.catpca_vars, "CATPCA Variables")

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        self.catpca_k = QSpinBox(); self.catpca_k.setRange(2, 10); self.catpca_k.setValue(2)
        form.addRow("Number of Components to Extract:", self.catpca_k)
        
        self.catpca_missing = QComboBox()
        self.catpca_missing.addItems(["Drop missing rows (Listwise)", "Impute missing as separate category"])
        form.addRow("Missing Value Handling:", self.catpca_missing)
        cgl.addLayout(form)

        btn = QPushButton("▶ Run CATPCA")
        btn.setStyleSheet("margin-top: 10px;")
        btn.clicked.connect(self.run_catpca)
        cgl.addWidget(btn)
        
        cl.addWidget(cg)

        tab, self.catpca_tabs = self.create_split_module(cw, bank=self.build_bank_panel())
        self.tabs.addWidget(tab)
        
    def run_catpca(self):
        if self.df is None: return
        if not PRINCE_AVAILABLE:
            self.catpca_tabs.addTab(
                QTextEdit("Missing dependency: please run  pip install prince  then restart PsyStat."),
                "Error"
            )
            return
        selected = [item.text() for item in self.catpca_vars.selectedItems()]
        if len(selected) < 3: return

        k = self.catpca_k.value()
        drop_missing = "Drop" in self.catpca_missing.currentText()

        try:
            if drop_missing:
                data = self.df[selected].dropna()
            else:
                data = self.df[selected].fillna("Missing")

            # ── Per-variable scaling level detection ──────────────────────────
            # Auto-detect: object/category → Nominal; int ≤10 unique → Ordinal;
            # otherwise → Numeric.  self.catpca_scaling_overrides (dict) allows
            # per-variable UI overrides (planned widget; defaults to empty dict).
            overrides = getattr(self, 'catpca_scaling_overrides', {})
            scaling = {}
            for col in selected:
                if col in overrides:
                    scaling[col] = overrides[col]
                elif data[col].dtype == object or str(data[col].dtype) == 'category':
                    scaling[col] = 'Nominal'
                elif pd.api.types.is_integer_dtype(data[col]) and data[col].nunique() <= 10:
                    scaling[col] = 'Ordinal'
                else:
                    scaling[col] = 'Numeric'

            num_cols = [c for c, lv in scaling.items() if lv == 'Numeric']
            cat_cols = [c for c, lv in scaling.items() if lv != 'Numeric']

            # ── Algorithm selection ───────────────────────────────────────────
            if len(num_cols) == 0:
                df_enc = data.copy()
                df_enc[cat_cols] = df_enc[cat_cols].astype(str)
                model = prince.MCA(n_components=k, n_iter=10, random_state=42)
                model = model.fit(df_enc)
                row_coords = model.row_coordinates(df_enc)
                col_coords = model.column_coordinates(df_enc)
                eigenvalues = list(model.eigenvalues_)
                explained   = list(model.percentage_of_variance_)
                method_label = "Multiple Correspondence Analysis (MCA)"
                df_fit = df_enc
            else:
                df_mixed = data.copy()
                df_mixed[cat_cols] = df_mixed[cat_cols].astype(str)
                model = prince.FAMD(n_components=k, n_iter=10, random_state=42)
                model = model.fit(df_mixed)
                row_coords = model.row_coordinates(df_mixed)
                col_coords = model.column_coordinates(df_mixed)
                eigenvalues = list(model.eigenvalues_)
                explained   = list(model.percentage_of_variance_)
                method_label = "Factor Analysis of Mixed Data (FAMD)"
                df_fit = df_mixed

            # ── Cronbach's Alpha per dimension ────────────────────────────────
            # Formula: α_d = (p/(p-1)) × (1 − p/λ_d)  (ten Berge & Hofstee, 1999)
            p = len(selected)
            cronbach_alphas = []
            for dim_col in row_coords.columns:
                scores = row_coords[dim_col].values
                lam = float(np.var(scores, ddof=0) * len(scores))
                if lam == 0 or p <= 1:
                    cronbach_alphas.append(float('nan'))
                else:
                    cronbach_alphas.append(round((p / (p - 1)) * (1.0 - p / lam), 4))

            timestamp = datetime.now().strftime("%H:%M:%S")
            out = self.get_apa_css() + "<h2>CATPCA Summary</h2>"
            out += f"<p><b>Method:</b> {method_label}</p>"

            # Variable scaling summary
            out += "<h3>Variable Scaling Levels</h3><table class='apa'><tr><th>Variable</th><th>Scaling Level</th></tr>"
            for col in selected:
                out += f"<tr><td style='text-align:left;'>{col}</td><td>{scaling[col]}</td></tr>"
            out += "</table>"

            # Model summary — eigenvalues + Cronbach's Alpha (matches SPSS Model Summary table)
            out += "<h3>Model Summary</h3><table class='apa'><tr><th>Dimension</th><th>Eigenvalue</th><th>% of Variance</th><th>Cumulative %</th><th>Cronbach's &alpha;</th></tr>"
            cum_var = 0.0
            for i in range(k):
                cum_var += explained[i]
                alpha_i = cronbach_alphas[i]
                alpha_str = self.fmt(alpha_i) if not (isinstance(alpha_i, float) and np.isnan(alpha_i)) else "—"
                alpha_color = "#EF4444" if isinstance(alpha_i, float) and not np.isnan(alpha_i) and alpha_i < 0 else "inherit"
                out += (f"<tr><td>Dimension {i+1}</td><td>{self.fmt(eigenvalues[i])}</td>"
                        f"<td>{self.fmt(explained[i])}%</td><td>{self.fmt(cum_var)}%</td>"
                        f"<td style='color:{alpha_color};'>{alpha_str}</td></tr>")
            out += "</table>"
            out += ("<div class='interpret'><i>Note.</i> Cronbach's &alpha; per dimension follows ten Berge &amp; Hofstee (1999): "
                    "&alpha; = (p/(p&minus;1)) &times; (1 &minus; p/&lambda;). A negative value means the eigenvalue is smaller than "
                    "the number of variables — that dimension should not be retained. Matches SPSS CATPCA Model Summary.</div>")

            # Category Quantifications (column coordinates) — SPSS equivalent
            out += "<h3>Category Quantifications</h3><table class='apa'><tr><th>Variable / Category</th>"
            for i in range(k): out += f"<th>Dimension {i+1}</th>"
            out += "</tr>"
            for idx_label, coord_row in col_coords.iterrows():
                out += f"<tr><td style='text-align:left;'>{idx_label}</td>"
                for i in range(k):
                    val = coord_row.iloc[i] if i < len(coord_row) else np.nan
                    bold_val = f"<b>{self.fmt(val)}</b>" if not np.isnan(float(val)) and abs(float(val)) > 0.4 else self.fmt(val)
                    out += f"<td>{bold_val}</td>"
                out += "</tr>"
            out += "</table><div class='interpret'><i>Note.</i> Category Quantifications are the optimal numeric values assigned to each category, equivalent to SPSS CATPCA \"Category Quantifications\". Loadings &ge; .40 are bolded.</div>"

            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
            self.catpca_tabs.addTab(tv, f"Results ({timestamp})")

            # Biplot using column (category) coordinates for Dim 1 vs Dim 2
            if MATPLOTLIB_AVAILABLE and k >= 2:
                fig = Figure(figsize=(7, 6))
                ax = fig.add_subplot(111)
                if self.is_dark_mode:
                    fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#374151')
                    ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                    ax.title.set_color('white'); ax.tick_params(colors='white')

                ax.axhline(0, color='#9CA3AF', linestyle='--')
                ax.axvline(0, color='#9CA3AF', linestyle='--')

                # Plot object scores (individuals) as small grey dots
                ax.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1],
                           color='#9CA3AF', s=12, alpha=0.4, zorder=1)

                # Plot category quantifications as labelled arrows
                for idx_label, coord_row in col_coords.iterrows():
                    x, y = float(coord_row.iloc[0]), float(coord_row.iloc[1])
                    ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.05,
                             fc='#4F46E5', ec='#4F46E5', alpha=0.7, zorder=2)
                    ax.text(x * 1.12, y * 1.12, str(idx_label),
                            color='#818CF8' if self.is_dark_mode else '#1E3A8A',
                            ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

                all_vals = np.abs(col_coords.values.astype(float))
                limit = float(np.nanmax(all_vals)) + 0.3 if all_vals.size > 0 else 1.5
                ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
                ax.set_xlabel(f"Dimension 1 ({self.fmt(explained[0])}% variance)")
                ax.set_ylabel(f"Dimension 2 ({self.fmt(explained[1])}% variance)")
                ax.set_title(f"CATPCA Biplot — {method_label}")
                fig.tight_layout()
                chart = self.make_zoomable_chart(fig)
                self.catpca_tabs.addTab(chart, f"Biplot ({timestamp})")

            self.catpca_tabs.setCurrentIndex(self.catpca_tabs.count() - 1)
        except Exception as e:
            self.catpca_tabs.addTab(QTextEdit(str(e)), "Error")


    # ==========================================
    # MODULE 6.2: CLUSTER & LATENT CLASS ANALYSIS
    # ==========================================
    def init_lca_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        if not SKLEARN_AVAILABLE:
            lbl = QLabel("Missing scikit-learn module. Cannot run cluster analysis.")
            lbl.setWordWrap(True); lbl.setMinimumWidth(10)
            cl.addWidget(lbl)
            self.tabs.addWidget(cw); return

        self.add_info_box(cl, "<b>Latent Class / Profile Analysis:</b> Discover hidden categorical subgroups (classes/profiles) in your population based on continuous variables using Gaussian Mixture Modeling (LPA) or standard K-Means.")

        lg = QGroupBox("Configuration")
        lg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        lgl = QVBoxLayout(lg)
        lgl.setSpacing(5)

        self.lca_vars = QListWidget(); self.lca_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lca_vars.setMinimumHeight(150)
        self.setup_list_selection(lgl, "Variables (Indicators):", self.lca_vars, "Indicators")
        
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        self.lca_method = QComboBox()
        self.lca_method.addItems(["Latent Profile Analysis (Gaussian Mixture)", "K-Means Clustering", "Hierarchical Clustering"])
        self.lca_k = QSpinBox(); self.lca_k.setRange(2, 20); self.lca_k.setValue(3)
        
        form.addRow("Extraction Method:", self.lca_method)
        form.addRow("Number of Classes/Clusters (k):", self.lca_k)
        lgl.addLayout(form)
        
        btn = QPushButton("▶ Run Latent Profile / Cluster Analysis")
        btn.setStyleSheet("margin-top: 10px;")
        btn.clicked.connect(self.run_lca)
        lgl.addWidget(btn)
        
        cl.addWidget(lg)
        
        tab, self.lca_tabs = self.create_split_module(cw, bank=self.build_bank_panel())
        self.tabs.addWidget(tab)

    def run_lca(self):
        if self.df is None or not SKLEARN_AVAILABLE: return
        selected = [item.text() for item in self.lca_vars.selectedItems()]
        if len(selected) < 2: return
        
        method = self.lca_method.currentText()
        k = self.lca_k.value()
        
        try:
            data = self.df[selected].dropna()
            X = (data - data.mean()) / data.std()
            
            if "K-Means" in method:
                model = KMeans(n_clusters=k, random_state=42)
                labels = model.fit_predict(X)
            elif "Hierarchical" in method:
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X)
            else:
                model = GaussianMixture(n_components=k, random_state=42)
                labels = model.fit_predict(X)
                
            data['Class'] = labels + 1
            class_means = data.groupby('Class').mean()
            counts = data['Class'].value_counts().sort_index()
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            out = self.get_apa_css() + f"<h2>{method} Results</h2>"
            
            if "Latent Profile Analysis" in method:
                entropy = np.nan
                try:
                    probs = model.predict_proba(X)
                    entropy = 1 - np.sum(-probs * np.log(probs + 1e-9)) / (len(X) * np.log(k))
                except: pass
                
                out += "<h3>Model Fit (Information Criteria)</h3><table class='apa'><tr><th>Metric</th><th>Value</th></tr>"
                out += f"<tr><td style='text-align:left;'>Akaike Information Criterion (AIC)</td><td>{self.fmt(model.aic(X))}</td></tr>"
                out += f"<tr><td style='text-align:left;'>Bayesian Information Criterion (BIC)</td><td>{self.fmt(model.bic(X))}</td></tr>"
                if not pd.isna(entropy):
                    out += f"<tr><td style='text-align:left;'>Entropy</td><td>{self.fmt(entropy)}</td></tr>"
                out += "</table>"
                out += "<div class='interpret'><i>Note.</i> <b>Best Practices for LCA/LPA (Weller et al., 2020):</b><br>"
                out += "<b>1. Model Fit:</b> Compare models with <i>k</i> and <i>k-1</i> classes. Lower AIC and BIC values indicate a better-fitting parsimonious model.<br>"
                out += "<b>2. Entropy:</b> A measure of classification uncertainty. Values &ge; .80 indicate good class separation.<br>"
                out += "<b>3. Class Size:</b> Avoid extracting classes that contain &lt; 5% of the sample, as they may be spurious.<br>"
                out += "<b>4. Interpretability:</b> Ensure the resulting profiles are theoretically meaningful.</div>"
            
            out += "<h3>Class/Cluster Sizes</h3><table class='apa'><tr><th>Class</th><th>Count</th><th>Percent</th></tr>"
            for c, cnt in counts.items():
                out += f"<tr><td>Class {c}</td><td>{cnt}</td><td>{self.fmt((cnt/len(data))*100)}%</td></tr>"
            out += "</table>"
            
            out += "<h3>Class Profiles (Raw Means)</h3><table class='apa'><tr><th>Class</th>"
            for col in selected: out += f"<th>{col}</th>"
            out += "</tr>"
            for c in class_means.index:
                out += f"<tr><td><b>Class {c}</b></td>"
                for col in selected: out += f"<td>{self.fmt(class_means.loc[c, col])}</td>"
                out += "</tr>"
            out += "</table><div class='interpret'><i>Note.</i> <b>Interpretation:</b> Class profiles allow you to see the defining characteristics (average scores) of each hidden subgroup. Data was standardized internally for extraction.</div>"
            
            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
            self.lca_tabs.addTab(tv, f"LCA/Profiles ({timestamp})")
            
            if MATPLOTLIB_AVAILABLE:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                fig = Figure(figsize=(7,5))
                ax = fig.add_subplot(111)
                if self.is_dark_mode:
                    fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#374151'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white'); ax.title.set_color('white'); ax.tick_params(colors='white')
                
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.set_title("Latent Classes Visualization (PCA Reduced)")
                
                legend1 = ax.legend(*scatter.legend_elements(), title="Classes", facecolor='#374151' if self.is_dark_mode else 'white', labelcolor='white' if self.is_dark_mode else 'black')
                ax.add_artist(legend1)
                
                fig.tight_layout()
                chart = self.make_zoomable_chart(fig)
                self.lca_tabs.addTab(chart, f"Plot ({timestamp})")

            self.lca_tabs.setCurrentIndex(self.lca_tabs.count() - 1)
        except Exception as e:
            self.lca_tabs.addTab(QTextEdit(str(e)), "Error")


    # ==========================================
    # MODULE 6.5: FORECASTING & GROWTH CURVES
    # ==========================================
    def init_forecast_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        cl.setSpacing(5) 
        
        self.add_info_box(cl, "<b>Forecasting & Growth Curves:</b> Predict future values or extract longitudinal growth trajectories.")

        self.fore_method_combo = QComboBox()
        self.fore_method_combo.addItems([
            "Exponential Smoothing (Single Series)",
            "Latent Growth Curve Modeling (LGCM)"
        ])
        
        lbl_method = QLabel("Method:")
        lbl_method.setWordWrap(True); lbl_method.setMinimumWidth(10)
        cl.addWidget(lbl_method)
        cl.addWidget(self.fore_method_combo)
        
        self.fore_stack = QStackedWidget()
        
        # 1. ES Widget
        w_es = QWidget(); l_es = QFormLayout(w_es)
        l_es.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        l_es.setContentsMargins(0,0,0,0)
        self.fore_t = QComboBox()
        self.fore_y = QComboBox()
        self.fore_steps = QSpinBox(); self.fore_steps.setRange(1, 100); self.fore_steps.setValue(10)
        l_es.addRow("Time/Date Variable (Optional):", self.fore_t)
        l_es.addRow("Target Variable to Forecast:", self.fore_y)
        l_es.addRow("Periods to Forecast:", self.fore_steps)
        
        # 2. LGCM Widget
        w_lgcm = QWidget(); l_lgcm = QVBoxLayout(w_lgcm)
        l_lgcm.setContentsMargins(0,0,0,0)
        self.lgcm_vars = QListWidget(); self.lgcm_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lgcm_vars.setMinimumHeight(120)
        self.setup_list_selection(l_lgcm, "Select Repeated Measures (T1, T2, T3... in order):", self.lgcm_vars, "LGCM Measures")
        
        self.fore_stack.addWidget(w_es)
        self.fore_stack.addWidget(w_lgcm)
        
        self.fore_method_combo.currentIndexChanged.connect(self.fore_stack.setCurrentIndex)
        cl.addWidget(self.fore_stack)

        btn = QPushButton("▶ Run Model")
        btn.setStyleSheet("margin-top: 15px;")
        btn.clicked.connect(self.run_forecast)
        cl.addWidget(btn)
        cl.addStretch()
        
        tab, self.fore_tabs = self.create_split_module(cw)
        self.tabs.addWidget(tab)

    def run_forecast(self):
        if self.df is None: return
        
        if self.fore_method_combo.currentText() == "Latent Growth Curve Modeling (LGCM)":
            self.run_lgcm()
            return
            
        t_var = self.fore_t.currentText()
        y_var = self.fore_y.currentText()
        steps = self.fore_steps.value()
        if not y_var: return
        
        try:
            data = self.df[y_var].dropna().values
            model = ExponentialSmoothing(data, trend='add', seasonal=None, initialization_method="estimated")
            fit = model.fit()
            forecast = fit.forecast(steps)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            out = self.get_apa_css() + "<h2>Exponential Smoothing Forecast Model</h2>"
            out += "<table class='apa'><tr><th>Metric</th><th>Value</th></tr>"
            out += f"<tr><td style='text-align:left;'>AIC</td><td>{self.fmt(fit.aic)}</td></tr>"
            out += f"<tr><td style='text-align:left;'>BIC</td><td>{self.fmt(fit.bic)}</td></tr>"
            out += f"<tr><td style='text-align:left;'>Smoothing Level (α)</td><td>{self.fmt(fit.params.get('smoothing_level', np.nan))}</td></tr>"
            out += f"<tr><td style='text-align:left;'>Smoothing Trend (β)</td><td>{self.fmt(fit.params.get('smoothing_trend', np.nan))}</td></tr>"
            out += "</table><div class='interpret'><i>Note.</i> <b>Interpretation:</b> This algorithm uses Holt's linear trend method (identical to base defaults in SPSS/Minitab) to project future data points by analyzing recent levels and trends. Lower AIC/BIC values indicate a better fitting model.</div>"
            
            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
            self.fore_tabs.addTab(tv, f"Forecast Stats ({timestamp})")
            
            if MATPLOTLIB_AVAILABLE:
                fig = Figure(figsize=(8,4))
                ax = fig.add_subplot(111)
                if self.is_dark_mode:
                    fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#374151'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white'); ax.title.set_color('white'); ax.tick_params(colors='white')
                
                hist_x = np.arange(len(data))
                fore_x = np.arange(len(data), len(data) + steps)
                
                ax.plot(hist_x, data, label="Historical Data", color='#CBD5E1' if self.is_dark_mode else 'black')
                ax.plot(fore_x, forecast, label="Forecast", color='#EF4444', linestyle='--', marker='o', markersize=4)
                ax.set_title(f"Forecast of {y_var} ({steps} periods)")
                ax.legend(facecolor='#374151' if self.is_dark_mode else 'white', labelcolor='white' if self.is_dark_mode else 'black')
                fig.tight_layout()
                chart = self.make_zoomable_chart(fig)
                self.fore_tabs.addTab(chart, f"Forecast Plot ({timestamp})")

            self.fore_tabs.setCurrentIndex(self.fore_tabs.count() - 1)
        except Exception as e:
            self.fore_tabs.addTab(QTextEdit(str(e)), "Error")

    def run_lgcm(self):
        if not SEM_AVAILABLE:
            QMessageBox.critical(self, "Error", "semopy is required for LGCM.")
            return
            
        vars = [item.text() for item in self.lgcm_vars.selectedItems()]
        if len(vars) < 3:
            QMessageBox.warning(self, "Warning", "Select at least 3 repeated measures for LGCM.")
            return
            
        try:
            data = self.df[vars].dropna()
            if data.empty:
                raise ValueError("Dataset is empty after dropping missing values.")
                
            i_parts = [f"1*{v}" for v in vars]
            s_parts = [f"{idx}*{v}" for idx, v in enumerate(vars)]
            syn = f"i =~ {' + '.join(i_parts)}\ns =~ {' + '.join(s_parts)}"
            
            model = ModelMeans(syn)
            model.fit(data)
            stats = calc_stats(model)
            ins = model.inspect()
            
            std_model = ModelMeans(syn)
            df_std = (data - data.mean()) / data.std().replace(0, 1)
            std_model.fit(df_std)
            std_ins = std_model.inspect()
            
            html = self.generate_sem_html_report(model, stats, ins, std_ins, len(data))
            html += self.generate_standardized_residuals_html(model)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(html)
            self.fore_tabs.addTab(tv, f"LGCM Results ({timestamp})")
            
            if MATPLOTLIB_AVAILABLE:
                fig = Figure(figsize=(7, 5))
                ax = fig.add_subplot(111)
                if self.is_dark_mode:
                    fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#374151')
                    ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                    ax.title.set_color('white'); ax.tick_params(colors='white')
                
                # Robust extraction of Intercepts for Plotting
                int_i_row = ins[(ins['lval'] == 'i') & (ins['op'] == '~1')]
                int_s_row = ins[(ins['lval'] == 's') & (ins['op'] == '~1')]
                
                int_i = int_i_row['Estimate'].values[0] if not int_i_row.empty else 0
                int_s = int_s_row['Estimate'].values[0] if not int_s_row.empty else 0
                
                time_pts = np.arange(len(vars))
                implied = int_i + time_pts * int_s
                obs = data.mean().values
                
                ax.plot(time_pts, obs, 'o-', color='#4F46E5', label="Observed Means", linewidth=2, markersize=8)
                ax.plot(time_pts, implied, 's--', color='#EF4444', label="Implied Growth Trajectory", linewidth=2.5)
                
                ax.set_xticks(time_pts)
                ax.set_xticklabels(vars)
                ax.set_ylabel("Score / Mean")
                ax.set_title("Latent Growth Curve Trajectory", fontweight="bold")
                ax.legend(facecolor='#374151' if self.is_dark_mode else 'white', labelcolor='white' if self.is_dark_mode else 'black')
                
                fig.tight_layout()
                chart = self.make_zoomable_chart(fig)
                self.fore_tabs.addTab(chart, f"Growth Plot ({timestamp})")

            self.fore_tabs.setCurrentIndex(self.fore_tabs.count() - 2)
            
        except Exception as e:
            self.fore_tabs.addTab(QTextEdit(str(e)), "Error")


    # ==========================================
    # MODULE 7: EFA
    # ==========================================
    def init_efa_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        if not FA_AVAILABLE: 
            lbl = QLabel("Missing module: factor_analyzer")
            lbl.setWordWrap(True); lbl.setMinimumWidth(10)
            cl.addWidget(lbl)
            self.tabs.addWidget(cw); return

        self.add_info_box(cl, "<b>Exploratory Factor Analysis (EFA):</b> A dimension-reduction technique used to uncover the underlying structure of a large set of variables. It helps you group survey items into 'Factors' based on how strongly they correlate with each other.")


        self.efa_vars = QListWidget(); self.efa_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.efa_vars.setMinimumHeight(150)
        self.setup_list_selection(cl, "Select Items for EFA:", self.efa_vars, "EFA Items")

        lbl_ext = QLabel("Extraction Method:")
        lbl_ext.setWordWrap(True); lbl_ext.setMinimumWidth(10)
        cl.addWidget(lbl_ext)
        self.efa_ext = QComboBox(); self.efa_ext.addItems(["minres (MINRES / PAF)", "ml (Maximum Likelihood)", "principal (PCA)"])
        cl.addWidget(self.efa_ext)
        
        lbl_rot = QLabel("Rotation Method:")
        lbl_rot.setWordWrap(True); lbl_rot.setMinimumWidth(10)
        cl.addWidget(lbl_rot)
        self.efa_rot = QComboBox(); self.efa_rot.addItems(["promax", "oblimin", "varimax", "none"])
        cl.addWidget(self.efa_rot)
        
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        self.efa_fac = QSpinBox(); self.efa_fac.setRange(0, 20)
        self.efa_sup = QDoubleSpinBox(); self.efa_sup.setRange(0.0, 0.99); self.efa_sup.setValue(0.30); self.efa_sup.setSingleStep(0.05)
        form.addRow("Number of Factors (0 = Auto Kaiser):", self.efa_fac)
        form.addRow("Suppress Loadings < :", self.efa_sup)
        cl.addLayout(form)
        
        btn = QPushButton("▶ Run EFA"); btn.clicked.connect(self.run_efa)
        btn.setStyleSheet("margin-top: 15px;")
        cl.addWidget(btn)
        
        tab, self.efa_tabs = self.create_split_module(cw, bank=self.build_bank_panel(numeric_only=True))
        self.tabs.addWidget(tab)

    def run_efa(self):
        if not FA_AVAILABLE or self.df is None: return
        items = [i.text() for i in self.efa_vars.selectedItems()]
        if len(items) < 3: return
        
        try:
            df_efa = self.df[items].dropna()
            data = df_efa.to_numpy() 
            _, kmo = calculate_kmo(data)
            chi, p_val = calculate_bartlett_sphericity(data)
            
            n_fac = self.efa_fac.value()
            ext = self.efa_ext.currentText().split(" ")[0]
            rot = self.efa_rot.currentText() if self.efa_rot.currentText() != "none" else None
            sup = self.efa_sup.value()
            
            tmp = FactorAnalyzer(rotation=None, method=ext); tmp.fit(data)
            ev, _ = tmp.get_eigenvalues()
            if n_fac == 0: n_fac = sum(ev > 1)
                
            fa = FactorAnalyzer(n_factors=n_fac, rotation=rot, method=ext)
            fa.fit(data)
            loadings = pd.DataFrame(fa.loadings_, index=items, columns=[f"Factor {i+1}" for i in range(n_fac)])
            
            # --- 1. Assumption & Scree Plot Merge ---
            out1 = self.get_apa_css() + "<h2>Analysis Summary</h2>"
            out1 += self.build_missing_data_note(self.df, items, len(df_efa))
            out1 += f"<p><b>Items Analyzed:</b> {len(items)}<br><b>Extraction Method:</b> {self.efa_ext.currentText()}<br><b>Rotation Method:</b> {self.efa_rot.currentText()}<br><b>Factors Extracted:</b> {n_fac}</p>"
            out1 += self.build_sample_size_warning(len(df_efa), "factor")
            
            out1 += "<h2>1. Assumption Checks</h2>"
            kmo_stat = "Pass" if kmo >= 0.60 else "<span class='warn'>Violated</span>"
            out1 += f"<p><b>KMO (Sampling Adequacy):</b> {self.fmt(kmo)} (Ideal > 0.60) - {kmo_stat}</p>"
            out1 += f"<p><b>Bartlett's Test of Sphericity:</b> Approx. Chi-Square = {self.fmt(chi)}, p = {self.fmt(p_val, True)}</p>"
            
            var_exp = fa.get_factor_variance() 
            out1 += "<h2>2. Total Variance Explained</h2><table class='apa'><tr><th>Factor</th><th>Eigenvalue</th><th>% of Variance</th><th>Cumulative %</th></tr>"
            for i in range(n_fac):
                out1 += f"<tr><td>Factor {i+1}</td><td>{self.fmt(ev[i])}</td><td>{self.fmt(var_exp[1][i]*100)}%</td><td>{self.fmt(var_exp[2][i]*100)}%</td></tr>"
            out1 += "</table>"
            
            w_assump = QWidget(); l_assump = QVBoxLayout(w_assump)
            tv1 = QTextEdit(); tv1.setReadOnly(True); tv1.setHtml(out1)
            
            splitter = QSplitter(Qt.Orientation.Vertical)
            splitter.addWidget(tv1)
            
            if MATPLOTLIB_AVAILABLE:
                fig = Figure(figsize=(7,3))
                ax = fig.add_subplot(111)
                if self.is_dark_mode:
                    fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#374151')
                    ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                    ax.title.set_color('white'); ax.tick_params(colors='white')
                
                x_factors = np.arange(1, len(ev) + 1)
                ax.plot(x_factors, ev, 'o-', color='#4F46E5', linewidth=2, markersize=8)
                ax.axhline(y=1, color='#EF4444', linestyle='--', linewidth=1.5, label='Eigenvalue = 1')
                ax.set_title("Scree Plot", fontweight='bold')
                ax.set_xlabel("Component / Factor Number")
                ax.set_ylabel("Eigenvalue")
                ax.set_xticks(x_factors)
                ax.legend(facecolor='#374151' if self.is_dark_mode else 'white', labelcolor='white' if self.is_dark_mode else 'black')
                
                fig.tight_layout()
                chart = self.make_zoomable_chart(fig)
                splitter.addWidget(chart)
            
            l_assump.addWidget(splitter)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.efa_tabs.addTab(w_assump, f"Assumptions & Scree ({timestamp})")
            
            # --- 2. Factor Matrix & Correlations Merge ---
            mat = self.get_apa_css() + f"<h2>3. Factor Matrix & Communalities</h2>"
            mat += "<table class='apa'><tr><th>Item</th>"
            for i in range(n_fac): mat += f"<th>Factor {i+1}</th>"
            mat += "<th>h² (Communality)</th><th>Uniqueness</th></tr>"
            
            h2 = fa.get_communalities()
            uniq = fa.get_uniquenesses()
            
            for idx, item in enumerate(items):
                mat += f"<tr><td style='text-align:left;'><b>{item}</b></td>"
                for j in range(n_fac):
                    val = fa.loadings_[idx, j]
                    if abs(val) < sup: mat += "<td></td>"
                    elif abs(val) >= 0.4: mat += f"<td><b>{self.fmt(val)}</b></td>"
                    else: mat += f"<td>{self.fmt(val)}</td>"
                mat += f"<td>{self.fmt(h2[idx])}</td><td>{self.fmt(uniq[idx])}</td></tr>"
            mat += "</table>"
            mat += f"<div class='interpret'><i>Note.</i> <b>Interpretation Guidelines (Hair et al., 2010):</b> Loadings &lt; {sup} are suppressed. Factor loadings &ge; .40 are typically considered substantial indicators of the underlying latent construct and are bolded.<br><b>h² (Communality):</b> The proportion of an item's variance explained by the extracted factors.<br><b>Uniqueness:</b> The variance not shared with the factors.<br><b>Comparison to R (psych):</b> Minor differences (e.g., &plusmn;0.01) between these loadings and R's <code>psych::fa</code> are normal and result from different optimization algorithms and Kaiser normalization defaults.</div>"
            
            mat += "<h2>Internal Consistency (Reliability)</h2>"
            mat += "<table class='apa'><tr><th>Factor</th><th>Included Items</th><th>Cronbach's α</th><th>McDonald's ω</th></tr>"
            for col in loadings.columns:
                factor_items = loadings[abs(loadings[col]) >= sup].index.tolist()
                alpha = calc_cronbach_alpha(df_efa[factor_items]) if len(factor_items) > 1 else np.nan
                omega_val, _ = calc_mcdonalds_omega(df_efa[factor_items]) if len(factor_items) > 2 else (np.nan, None)
                alpha_cell = f"<span style='font-weight:900; color:#4338CA; font-size:16px;'>{self.fmt(alpha)}</span>"
                omega_cell = f"<span style='font-weight:900; color:#047857; font-size:16px;'>{self.fmt(omega_val)}</span>" if not np.isnan(omega_val) else "<span style='color:#9CA3AF;'>N/A (&lt;3 items)</span>"
                mat += f"<tr><td>{col}</td><td>{', '.join(factor_items) if factor_items else 'None'}</td><td>{alpha_cell}</td><td>{omega_cell}</td></tr>"
            mat += "</table>"
            mat += "<p style='font-size:12px; color:#6B7280;'><i>ω (McDonald, 1999) does not assume equal item loadings and is generally the more accurate reliability estimate when loadings vary across items.</i></p>"
            mat += ("<div style='background:#F3F4F6; border-left:4px solid #6B7280; padding:8px 12px; "
                    "margin:10px 0; font-size:12px; color:#374151;'>"
                    "<b>Comparison Note (SPSS Parity):</b> Extraction method <i>minres</i> (Minimum Residual) "
                    "is mathematically equivalent to SPSS's Principal Axis Factoring (PAF). Minor differences in "
                    "communalities or loadings at the 4th decimal place (typically &lt; 0.01) are expected due to "
                    "differences between the SLSQP optimiser used here and SPSS's proprietary iterative PAF routines. "
                    "These are <b>not errors</b> — they reflect convergence tolerance differences across platforms and "
                    "do not affect substantive interpretation."
                    "</div>")
            
            if n_fac > 1:
                mat += "<h2>4. Factor Correlation Matrix</h2>"
                if hasattr(fa, 'phi_') and fa.phi_ is not None:
                    phi = fa.phi_
                    mat += "<table class='apa'><tr><th></th>"
                    for i in range(n_fac): mat += f"<th>Factor {i+1}</th>"
                    mat += "</tr>"
                    for i in range(n_fac):
                        mat += f"<tr><td style='text-align:left;'><b>Factor {i+1}</b></td>"
                        for j in range(n_fac):
                            if i == j: mat += "<td>-</td>"
                            else: mat += f"<td>{self.fmt(phi[i, j])}</td>"
                        mat += "</tr>"
                    mat += "</table>"
                else:
                    mat += "<p style='color:#6B7280;'><i>Inter-factor correlations are assumed to be zero (0) because you used an Orthogonal rotation method (e.g., Varimax) or extracted factors without rotation. Use 'promax' or 'oblimin' rotation to allow factors to correlate with one another.</i></p>"

            tv2 = QTextEdit(); tv2.setReadOnly(True); tv2.setHtml(mat)
            self.efa_tabs.addTab(tv2, f"Matrix & Corr ({timestamp})")

            self.efa_tabs.setCurrentIndex(self.efa_tabs.count() - 1)
        except Exception as e:
            self.efa_tabs.addTab(QTextEdit(str(e)), "Error")


    # ==========================================
    # MODULE 8: CFA 
    # ==========================================
    def init_cfa_tab(self):
        cw = QWidget(); cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        if not SEM_AVAILABLE: 
            lbl = QLabel("Missing module: semopy")
            lbl.setWordWrap(True); lbl.setMinimumWidth(10)
            cl.addWidget(lbl)
            self.tabs.addWidget(cw); return

        self.add_info_box(cl, "<b>Confirmatory Factor Analysis (CFA):</b> Tests whether your data fits a pre-specified theoretical measurement model. You must declare exactly which items belong to which latent variables.")

        lbl_ex = QLabel("Example Syntax (Do not edit):")
        lbl_ex.setWordWrap(True); lbl_ex.setMinimumWidth(10)
        cl.addWidget(lbl_ex)
        
        ex_box = QTextEdit()
        ex_box.setReadOnly(True)
        ex_box.setStyleSheet("background-color: #F3F4F6; color: #4B5563; font-family: Consolas;")
        if self.is_dark_mode: ex_box.setStyleSheet("background-color: #1F2937; color: #9CA3AF; font-family: Consolas;")
        ex_box.setPlainText("# Use '=~' to assign indicators to a Latent factor\nAnxiety =~ item1 + item2 + item3\nDepression =~ item4 + item5 + item6")
        # Size the box to fit its actual 3-line example exactly, rather than letting a
        # generic QTextEdit expand and eat space the syntax editor below needs more.
        fm = ex_box.fontMetrics()
        line_count = ex_box.toPlainText().count('\n') + 1
        fitted_height = int(fm.lineSpacing() * line_count + 18)  # small padding for margins/scrollbar headroom
        ex_box.setFixedHeight(fitted_height)
        ex_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        ex_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cl.addWidget(ex_box)

        lbl_syn = QLabel("Your Lavaan Syntax:")
        lbl_syn.setWordWrap(True); lbl_syn.setMinimumWidth(10)
        cl.addWidget(lbl_syn)
        
        self.cfa_syntax = QTextEdit()
        self.cfa_syntax.setFont(QFont("Consolas", 12))
        self.cfa_syntax.setMinimumHeight(150)
        cl.addWidget(self.cfa_syntax)
        btn = QPushButton("▶ Run CFA"); btn.clicked.connect(self.run_cfa)
        btn.setStyleSheet("margin-top: 15px;")
        cl.addWidget(btn)
        
        tab, self.cfa_tabs = self.create_split_module(cw)
        self.tabs.addWidget(tab)

    def run_cfa(self):
        if not SEM_AVAILABLE or self.df is None: return
        syn = self.cfa_syntax.toPlainText().strip()
        if not syn: return
        
        import re
        syn = re.sub(r'(?<![~<>=!])=(?![~<>=])', '=~', syn)
        
        try:
            num_df = self.df.select_dtypes(include=[np.number]).dropna()
            if num_df.empty:
                raise ValueError("Dataset has no numeric columns or no rows left after dropping missing values.")
            
            lat_vars = []
            for line in syn.split('\n'):
                if '=~' in line:
                    lat_name = line.split('=~')[0].strip()
                    if lat_name and lat_name not in lat_vars:
                        lat_vars.append(lat_name)
            
            model = Model(syn)
            model.fit(num_df)
            stats = calc_stats(model)

            # Unstandardized inspect — used for parameter table and reliability
            ins     = model.inspect()
            std_ins = pd.DataFrame()

            html = self.generate_sem_html_report(model, stats, ins, std_ins, len(num_df))

            # ── Standardized loadings for reliability/AVE ─────────────────────
            # We need Std.all (fully standardized) loadings for CR and AVE.
            # Try sources in order of accuracy, same hierarchy as the report generator.
            std_load_src = None   # will be a DataFrame with lval/op/rval/std_col
            std_col      = None

            try:
                _ins_all = model.inspect(std_est=True)
                if 'Est. Std' in _ins_all.columns:
                    std_load_src = _ins_all
                    std_col      = 'Est. Std'
            except Exception:
                pass

            if std_load_src is None:
                # Manual fallback: compute Std.all from implied covariance matrix
                try:
                    lv_var = {r['lval']: float(r['Estimate'])
                              for _, r in ins[ins['op'] == '~~'].iterrows()
                              if r['lval'] == r['rval']}
                    sigma_raw = model.calculate_sigma()[0]
                    obs_cols  = [c for c in num_df.columns
                                 if c in ins['rval'].values or c in ins['lval'].values]
                    obs_cols  = obs_cols[:sigma_raw.shape[0]]
                    sigma     = pd.DataFrame(sigma_raw, index=obs_cols, columns=obs_cols)

                    def _sd(name):
                        if name in sigma.columns:
                            v = sigma.loc[name, name]
                            return float(np.sqrt(v)) if v > 0 else 1.0
                        return float(np.sqrt(lv_var.get(name, 1.0))) if lv_var.get(name, 1.0) > 0 else 1.0

                    rows_manual = ins.copy()
                    rows_manual['Est. Std'] = rows_manual.apply(
                        lambda r: float(r['Estimate']) * _sd(r['rval']) / _sd(r['lval'])
                        if _sd(r['lval']) != 0 else np.nan, axis=1
                    )
                    std_load_src = rows_manual
                    std_col      = 'Est. Std'
                except Exception:
                    std_load_src = ins   # last resort: use unstandardized
                    std_col      = 'Estimate'

            html += "<h2>Construct Reliability and Validity</h2>"
            html += ("<table class='apa'><tr>"
                     "<th>Latent Construct</th>"
                     "<th>Cronbach's &alpha;</th>"
                     "<th>McDonald's &omega; (= CR)</th>"
                     "<th>Average Variance Extracted (AVE)</th>"
                     "</tr>")

            for lat in lat_vars:
                try:
                    rows = std_load_src[
                        ((std_load_src['lval'] == lat) & (std_load_src['op'] == '=~')) |
                        ((std_load_src['rval'] == lat) & (std_load_src['op'] == '~'))
                    ]
                    std_loadings = pd.to_numeric(rows[std_col], errors='coerce').dropna().values

                    if len(std_loadings) > 0:
                        sum_lam      = np.sum(std_loadings)
                        sum_lam2     = np.sum(std_loadings ** 2)
                        error_vars   = 1 - (std_loadings ** 2)
                        sum_errors   = np.sum(error_vars)

                        cr  = (sum_lam ** 2) / ((sum_lam ** 2) + sum_errors) if (sum_lam ** 2 + sum_errors) > 0 else np.nan
                        ave = sum_lam2 / (sum_lam2 + sum_errors) if (sum_lam2 + sum_errors) > 0 else np.nan

                        # Cronbach's alpha from raw item scores
                        item_names  = rows['rval'].tolist() + rows['lval'].tolist()
                        valid_items = [i for i in item_names if i in num_df.columns and i != lat]
                        alpha       = calc_cronbach_alpha(num_df[valid_items]) if len(valid_items) > 1 else np.nan

                        omega_cell = (f"<span style='font-weight:900; color:#047857; font-size:16px;'>"
                                      f"{self.fmt(cr)}</span>")
                        html += (f"<tr><td>{lat}</td><td>{self.fmt(alpha)}</td>"
                                 f"<td>{omega_cell}</td><td>{self.fmt(ave)}</td></tr>")
                    else:
                        html += f"<tr><td>{lat}</td><td>-</td><td>-</td><td>-</td></tr>"
                except Exception:
                    html += f"<tr><td>{lat}</td><td>-</td><td>-</td><td>-</td></tr>"
                    
            html += "</table><div class='interpret'><i>Note.</i> <b>Guidelines (Fornell & Larcker, 1981):</b> Excellent construct validity is established when AVE ≥ 0.50 and CR ≥ 0.70. <b>CR (Composite Reliability) is computed identically to McDonald's ω (total)</b> from standardized factor loadings (ω = (Σλ)² / [(Σλ)² + Σθ]; McDonald, 1999) -- the two terms are used interchangeably in the SEM/CFA literature.</div>"
            
            # Appending Hu & Bentler 1999 Interpretation Guide for CFA Fit
            html += "<div class='interpret' style='margin-top:15px; border-top: 1px dashed #9CA3AF; padding-top: 10px;'><i>Note.</i> <b>Model Fit Interpretation Guidelines (Hu & Bentler, 1999):</b><br>"
            html += "<b>Excellent Fit:</b> CFI &ge; .95, TLI &ge; .95, RMSEA &le; .06, SRMR &le; .08.<br>"
            html += "<b>Acceptable Fit:</b> CFI/TLI &ge; .90, RMSEA &le; .08, SRMR &le; .10.<br>"
            html += "<b>Comparison to R (lavaan):</b> Minor differences in fit indices or standard errors compared to R's <code>lavaan</code> are mathematically normal. They stem from differences in the default optimization algorithms (SLSQP vs NLMINB).</div>"
            
            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(html)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.cfa_tabs.addTab(tv, f"CFA Results ({timestamp})")

            # Modification Indices as their own clearly-labeled, easy-to-find tab
            # rather than buried at the bottom of the main results -- this is the
            # "what should I add to improve fit?" diagnostic researchers look for.
            mi_html = self.get_apa_css() + self.generate_standardized_residuals_html(model)
            mi_tv = QTextEdit(); mi_tv.setReadOnly(True); mi_tv.setHtml(mi_html)
            self.cfa_tabs.addTab(mi_tv, f"Modification Indices ({timestamp})")

            self.cfa_tabs.setCurrentWidget(tv)
        except Exception as e:
            self.cfa_tabs.addTab(QTextEdit(str(e)), "Error")


    # ==========================================
    # MODULE 9: SEM (Drag & Drop Canvas)
    # ==========================================
    def init_sem_tab(self):
        cw = QWidget(); cl = QVBoxLayout(cw)
        if not SEM_AVAILABLE: 
            lbl = QLabel("Requires semopy module.")
            lbl.setWordWrap(True); lbl.setMinimumWidth(10)
            cl.addWidget(lbl)
            self.tabs.addWidget(cw); return
        
        self.add_info_box(cl, "<b>Structural Equation Modeling (SEM):</b> Build full structural models. Use the visual builder or type syntax directly.")

        lbl_syn = QLabel("Lavaan Syntax:")
        lbl_syn.setWordWrap(True); lbl_syn.setMinimumWidth(10)
        cl.addWidget(lbl_syn)
        
        self.sem_syntax = QTextEdit(); self.sem_syntax.setFont(QFont("Consolas", 14)); cl.addWidget(self.sem_syntax)
        
        btn_run = QPushButton("▶ Estimate Structural Model")
        btn_run.setStyleSheet("margin-top: 15px;")
        btn_run.clicked.connect(self.run_sem)
        cl.addWidget(btn_run)
        
        tab, self.sem_out_tabs = self.create_split_module(cw)
        
        self.sem_tabs = QTabWidget()
        
        text_w = QWidget(); split_l = QVBoxLayout(text_w); split_l.setContentsMargins(0,0,0,0)
        split_l.addWidget(tab)
        self.sem_tabs.addTab(text_w, "Syntax & Results")
        
        canvas_w = QWidget(); cl_canvas = QVBoxLayout(canvas_w)
        
        guide = QLabel("<b>Interactive SEM Builder Guide:</b><br>"
                       "1. <b>Add Factors (Latent):</b> Click '✚ Add Latent', enter a name (no spaces).<br>"
                       "2. <b>Add Data Columns (Observed):</b> Select from dropdown and click '✚ Add Observed'.<br>"
                       "3. <b>Draw Paths:</b> Click '➚ Draw Path', click the <b>Source</b> node, then click the <b>Target</b> node.<br>"
                       "4. <b>Build Model:</b> Click 'Generate Syntax ↓' to convert your drawing into Lavaan syntax.")
        guide.setStyleSheet("background-color: #EEF2FF; border-left: 4px solid #4F46E5; padding: 10px; margin-bottom: 10px; color: #1E3A8A;")
        if self.is_dark_mode:
            guide.setStyleSheet("background-color: #1E293B; border-left: 4px solid #818CF8; padding: 10px; margin-bottom: 10px; color: #E2E8F0;")
        guide.setWordWrap(True); guide.setMinimumWidth(10) # Fluid Wrap Trick
        cl_canvas.addWidget(guide)
        
        tb_outer = QVBoxLayout()

        # --- Add Latent group: its own labeled row, inline name entry (no blocking
        # modal dialog) so the user can see the canvas and existing node names while
        # typing, instead of a popup that hides everything else. ---
        latent_row = QHBoxLayout()
        latent_label = QLabel("Add Latent Factor:")
        latent_label.setStyleSheet("font-weight:bold; min-width:130px;")
        self.sem_latent_name = QLineEdit()
        self.sem_latent_name.setPlaceholderText("Type a new factor name (no spaces)…")
        btn_l = QPushButton("✚ Add Latent")
        btn_l.clicked.connect(self.add_latent_from_field)
        self.sem_latent_name.returnPressed.connect(self.add_latent_from_field)
        latent_row.addWidget(latent_label)
        latent_row.addWidget(self.sem_latent_name)
        latent_row.addWidget(btn_l)
        tb_outer.addLayout(latent_row)

        # --- Add Observed group: its own labeled row with the existing dataset-column
        # dropdown, kept in sync via update_global_dropdowns like every other combo
        # in the app, and drag-and-drop enabled for consistency with the rest of the UI. ---
        observed_row = QHBoxLayout()
        observed_label = QLabel("Add Observed Variable:")
        observed_label.setStyleSheet("font-weight:bold; min-width:130px;")
        self.sem_var_combo = QComboBox()
        self.sem_var_combo.setPlaceholderText("Select a column from your dataset…")
        self.enable_drag_drop_combo(self.sem_var_combo)
        btn_o = QPushButton("✚ Add Observed")
        btn_o.clicked.connect(self.add_observed_from_combo)
        observed_row.addWidget(observed_label)
        observed_row.addWidget(self.sem_var_combo)
        observed_row.addWidget(btn_o)
        tb_outer.addLayout(observed_row)

        tb = QHBoxLayout(); self.sem_canvas = SEMCanvas()

        btn_p = QPushButton("➚ Draw Path")
        btn_p.setCheckable(True)
        btn_p.setStyleSheet("""
            QPushButton { background-color: #4F46E5; color: white; padding: 10px 15px; border-radius: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #4338CA; }
            QPushButton:checked { background-color: #1E3A8A; border: 2px inset #1E40AF; }
        """)
        btn_p.clicked.connect(lambda checked: setattr(self.sem_canvas, 'mode', 'Path' if checked else 'Select'))
        
        btn_del = QPushButton("✖ Delete")
        btn_del.clicked.connect(self.sem_canvas.delete_selected)
        
        btn_example = QPushButton("Load Example")
        btn_example.clicked.connect(self.sem_canvas.load_example)
        
        btn_gen = QPushButton("Generate Syntax ↓"); btn_gen.setStyleSheet("background-color:#E53E3E;")
        btn_gen.clicked.connect(lambda: self.sem_syntax.setPlainText(self.sem_canvas.generate_syntax()))
        btn_gen.clicked.connect(lambda: self.sem_tabs.setCurrentIndex(0)) 
        
        tb.addWidget(btn_example); tb.addWidget(btn_p); tb.addWidget(btn_del); tb.addStretch(); tb.addWidget(btn_gen)
        tb_outer.addLayout(tb)
        cl_canvas.addLayout(tb_outer); cl_canvas.addWidget(self.sem_canvas)
        self.sem_tabs.addTab(canvas_w, "Interactive Builder")

        self.tabs.addWidget(self.sem_tabs)

    def add_latent_from_field(self):
        name = self.sem_latent_name.text().strip().replace(" ", "")
        if name:
            self.sem_canvas.add_node(name, is_latent=True)
            self.sem_latent_name.clear()

    def add_observed_from_combo(self):
        name = self.sem_var_combo.currentText()
        if name:
            self.sem_canvas.add_node(name, is_latent=False)

    def run_sem(self):
        if not SEM_AVAILABLE or self.df is None: return
        syntax = self.sem_syntax.toPlainText().strip()
        if not syntax: return
        self.sem_out_tabs.clear()
        
        try:
            num_df = self.df.select_dtypes(include=[np.number]).dropna()
            model = Model(syntax)
            model.fit(num_df)
            stats = calc_stats(model)
            
            try:
                ins = model.inspect(std_est=True)
                has_native_std = 'Est. Std' in ins.columns
            except TypeError:
                ins = model.inspect()
                has_native_std = False
                
            if has_native_std:
                std_ins = pd.DataFrame()
            else:
                df_std = (num_df - num_df.mean()) / num_df.std().replace(0, 1)
                std_model = Model(syntax)
                std_model.fit(df_std)
                std_ins = std_model.inspect()
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            html = self.generate_sem_html_report(model, stats, ins, std_ins, len(num_df))
            html += self.generate_standardized_residuals_html(model)

            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(html)
            self.sem_out_tabs.addTab(tv, f"SEM Results ({timestamp})")
            self.sem_out_tabs.setCurrentWidget(tv)
            
        except Exception as e: 
            self.sem_out_tabs.addTab(QTextEdit(str(e)), "Error")


    # ==========================================
    # MODULE 10: NETWORK ANALYSIS (DRAGGABLE GGM)
    # ==========================================
    def init_sna_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        if not NX_AVAILABLE: 
            lbl = QLabel("<b>Missing networkx module.</b><br><br>Network Analysis requires the external 'networkx' library.<br>Please install it via your terminal or command prompt by running:<br><br><code>pip install networkx</code>")
            lbl.setWordWrap(True)
            lbl.setMinimumWidth(10)
            lbl.setStyleSheet("font-size: 14px;")
            cl.addWidget(lbl)
            self.tabs.addWidget(cw); return

        self.add_info_box(cl, "<b>Psychometric Network Analysis (SNA):</b> Map out complex systems by examining how variables (nodes) connect to each other. This computes a Gaussian Graphical Model (GGM) Network where edges represent strong partial correlations. (Green = Positive, Red = Negative).")

        self.sna_vars = QListWidget()
        self.sna_vars.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.sna_vars.setMinimumHeight(150)
        self.setup_list_selection(cl, "Select Variables for Network:", self.sna_vars, "Network Variables")

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows) # Corrected Fluid Layout Fix
        self.sna_threshold = QDoubleSpinBox()
        self.sna_threshold.setRange(0.0, 1.0)
        self.sna_threshold.setValue(0.15)
        self.sna_threshold.setSingleStep(0.05)
        form.addRow("Minimum Partial Correlation Threshold (Absolute |r|):", self.sna_threshold)
        
        self.sna_layout = QComboBox()
        self.sna_layout.addItems(["Spring Layout (Fruchterman-Reingold)", "Circular Layout", "Kamada-Kawai Layout"])
        form.addRow("Graph Layout:", self.sna_layout)
        
        btn = QPushButton("▶ Run Network Analysis")
        btn.clicked.connect(self.run_sna)
        btn.setStyleSheet("margin-top: 15px;")
        form.addRow(btn)
        
        cl.addLayout(form)
        
        tab, self.sna_tabs = self.create_split_module(cw, bank=self.build_bank_panel(numeric_only=True))
        self.tabs.addWidget(tab)

    def run_sna(self):
        if not NX_AVAILABLE or self.df is None: return
        variables = [item.text() for item in self.sna_vars.selectedItems()]
        if len(variables) < 2: return
        
        try:
            data = self.df[variables].dropna()
            
            # --- Calculating Gaussian Graphical Model (Partial Correlations) ---
            R = data.corr().values
            try:
                P = np.linalg.inv(R)
            except np.linalg.LinAlgError:
                P = np.linalg.pinv(R)
                
            partial_corr = np.zeros_like(R)
            for i in range(len(R)):
                for j in range(len(R)):
                    if i == j:
                        partial_corr[i,j] = 1.0
                    else:
                        partial_corr[i,j] = -P[i,j] / np.sqrt(P[i,i] * P[j,j])
            
            corr_matrix = pd.DataFrame(partial_corr, index=data.columns, columns=data.columns)
            # -------------------------------------------------------------------
            
            threshold = self.sna_threshold.value()

            G = nx.Graph()
            for v in variables:
                G.add_node(v)

            for i in range(len(variables)):
                for j in range(i+1, len(variables)):
                    v1, v2 = variables[i], variables[j]
                    weight = corr_matrix.loc[v1, v2]
                    if abs(weight) >= threshold:
                        G.add_edge(v1, v2, weight=weight)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            out = self.get_apa_css() + "<h2>Psychometric Network Statistics (GGM)</h2>"
            out += f"<p><b>Total Nodes (Variables):</b> {G.number_of_nodes()} | <b>Total Edges (Partial Corrs):</b> {G.number_of_edges()} | <b>Density:</b> {self.fmt(nx.density(G))}</p>"
            
            deg = nx.degree_centrality(G)
            bet = nx.betweenness_centrality(G)
            try: clo = nx.closeness_centrality(G)
            except: clo = {n: np.nan for n in G.nodes()}
            
            strength = {}
            for node in G.nodes():
                strength[node] = sum(abs(G[node][neighbor]['weight']) for neighbor in G[node])
            
            out += "<table class='apa'><tr><th>Node (Variable)</th><th>Node Strength (Influence)</th><th>Degree Centrality</th><th>Betweenness Centrality</th><th>Closeness Centrality</th></tr>"
            for node in G.nodes():
                out += f"<tr><td style='text-align:left;'><b>{node}</b></td><td>{self.fmt(strength[node])}</td><td>{self.fmt(deg[node])}</td><td>{self.fmt(bet[node])}</td><td>{self.fmt(clo[node])}</td></tr>"
            out += "</table><div class='interpret'><i>Note.</i> <b>Interpretation (Epskamp et al., 2012):</b><br>"
            out += "<b>Node Strength:</b> The sum of absolute correlation weights connected to a node. High strength means the item heavily influences the overall network.<br>"
            out += "<b>Degree:</b> Proportional to the number of direct connections.<br>"
            out += "<b>Betweenness:</b> How often a variable acts as a bridge along the shortest path between other variables. Identifies key gatekeepers.<br>"
            out += "<b>Closeness:</b> How close a node is to all other nodes (higher means faster influence spread).</div>"
            
            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
            self.sna_tabs.addTab(tv, f"Network Metrics ({timestamp})")
            
            if MATPLOTLIB_AVAILABLE:
                fig = Figure(figsize=(8, 7))
                ax = fig.add_subplot(111)
                
                layout_style = self.sna_layout.currentText()
                if "Circular" in layout_style: pos = nx.circular_layout(G)
                elif "Kamada" in layout_style: pos = nx.kamada_kawai_layout(G)
                else: pos = nx.spring_layout(G, seed=42, k=0.5/np.sqrt(G.number_of_nodes()))
                
                node_sizes = [500 + (strength.get(n, 0) * 400) for n in G.nodes()]

                # Inner function to allow dynamic redrawing
                def draw_network():
                    ax.clear()
                    if self.is_dark_mode:
                        fig.patch.set_facecolor('#1F2937')
                        ax.set_facecolor('#374151')
                        
                    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#818CF8' if self.is_dark_mode else '#93C5FD', node_size=node_sizes, edgecolors='white', linewidths=1.5)
                    
                    for u, v, d in G.edges(data=True):
                        w = d['weight']
                        c = '#10B981' if w > 0 else '#EF4444' # Green for Positive, Red for Negative GGM
                        s = 'solid' if w > 0 else 'dashed'
                        thick = max(1.0, abs(w) * 8)
                        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=c, linewidth=thick, linestyle=s, alpha=0.7, zorder=1)

                    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold', font_color='white' if self.is_dark_mode else 'black')

                    ax.set_title(f"GGM Network (Partial |r| > {threshold})\n* Click and drag nodes to reposition", color='white' if self.is_dark_mode else 'black', pad=10)
                    ax.axis('off')

                draw_network()
                fig.tight_layout()
                canvas = self.make_resizable_canvas(fig)
                
                # Setup Interactive Dragging
                canvas.dragging_node = None
                
                def on_press(event):
                    if event.inaxes != ax or event.xdata is None or event.ydata is None: return
                    min_dist = float('inf')
                    closest = None
                    for n, p in pos.items():
                        dist = np.hypot(p[0] - event.xdata, p[1] - event.ydata)
                        if dist < min_dist:
                            min_dist = dist
                            closest = n
                    if min_dist < 0.15: 
                        canvas.dragging_node = closest

                def on_motion(event):
                    if canvas.dragging_node is None or event.inaxes != ax or event.xdata is None: return
                    pos[canvas.dragging_node] = np.array([event.xdata, event.ydata])
                    draw_network()
                    canvas.draw_idle()

                def on_release(event):
                    canvas.dragging_node = None

                canvas.mpl_connect('button_press_event', on_press)
                canvas.mpl_connect('motion_notify_event', on_motion)
                canvas.mpl_connect('button_release_event', on_release)
                
                self.sna_tabs.addTab(canvas, f"Network Plot ({timestamp})")

            self.sna_tabs.setCurrentIndex(self.sna_tabs.count() - 1)
        except Exception as e:
            self.sna_tabs.addTab(QTextEdit(str(e)), "Error")

    # ==========================================
    # MODULE 11: POWER ANALYSIS
    # ==========================================
    def init_power_tab(self):
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.add_info_box(cl, "<b>Power Analysis:</b> 'A Priori' mode calculates how many participants you need before collecting data. 'Post-Hoc / Sensitivity' mode takes the sample size you already have and tells you either the power you actually achieved, or the smallest effect size you'd have been able to detect — the questions reviewers usually ask after data collection.")

        form_widget = QWidget()
        form_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        form_widget.setMaximumWidth(320)
        
        # QFormLayout with WrapLongRows: short labels sit beside their field as normal,
        # but a label too long to fit (e.g. "Total Predictors (incl. interaction)")
        # automatically wraps to its own line above the field instead of forcing the
        # whole label column to stay wide for every row. This is the same fluid-layout
        # pattern already used throughout the rest of the app.
        form = QFormLayout(form_widget)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(12)

        self.p_mode = QComboBox()
        self.p_mode.addItems([
            "A Priori: Find Required Sample Size (N)",
            "Post-Hoc: Find Achieved Power (given N)",
            "Sensitivity: Find Minimum Detectable Effect (given N and Power)",
        ])
        mode_tips = [
            "Use BEFORE collecting data, when you're planning a study: tells you how many participants to recruit to reliably detect an effect of the size you expect.",
            "Use AFTER collecting data, when you already have a sample and want to know how much power your completed (or already-published) study actually had to detect a given effect size.",
            "Use AFTER collecting data when you want to know the smallest true effect your study could have reliably detected — useful for interpreting a non-significant result (\"was my study just underpowered?\").",
        ]
        for i, tip in enumerate(mode_tips):
            self.p_mode.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
        self.p_mode.currentIndexChanged.connect(lambda i: self.p_mode.setToolTip(mode_tips[i] if 0 <= i < len(mode_tips) else ""))
        self.p_mode.setToolTip(mode_tips[0])
        
        self.p_test = QComboBox()
        self.p_test.addItems([
            "Independent T-Test", 
            "Paired T-Test", 
            "ANOVA: One-Way (Fixed Effects, Omnibus)",
            "ANOVA: Two-Way / Factorial (Main Effect or Interaction)",
            "Multiple Regression (Omnibus R²)",
            "Multiple Regression (Interaction / R² Increase)",
            "Logistic Regression (Single Continuous Predictor)",
            "Correlation (Pearson r)", 
            "Chi-Square Test",
            "Mediation (X→M→Y Indirect Effect)",
        ])
        test_tips = [
            "Use when comparing the means of two SEPARATE, unrelated groups (e.g. Treatment vs. Control, Men vs. Women) on one continuous outcome.",
            "Use when comparing two related measurements from the SAME subjects (e.g. pre-test vs. post-test, or matched pairs).",
            "Use when comparing means across 3+ independent groups on one continuous outcome with a single categorical factor (e.g. 3 diet types).",
            "Use when you have TWO categorical factors at once (e.g. Diet × Gender) and want power for a main effect or their interaction specifically.",
            "Use to power a regression model's OVERALL explanatory strength (the full-model R² test) with a given number of predictors.",
            "Use to power a SPECIFIC subset of predictors (e.g. just an interaction term) added on top of a larger regression model.",
            "Use for a logistic regression with ONE continuous predictor, when your outcome is binary (e.g. pass/fail, disease/no disease) and you want to detect a given odds ratio.",
            "Use when testing whether a Pearson correlation between two continuous variables is significantly different from zero.",
            "Use for tests of association between categorical variables (e.g. a goodness-of-fit or independence test on frequency counts).",
            "No power calculator available for this design (see note below) — shows guidance instead of a number.",
        ]
        for i, tip in enumerate(test_tips):
            self.p_test.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
        self.p_test.currentIndexChanged.connect(lambda i: self.p_test.setToolTip(test_tips[i] if 0 <= i < len(test_tips) else ""))
        self.p_test.setToolTip(test_tips[0])

        self.lbl_test_explain = QLabel(test_tips[0])
        self.lbl_test_explain.setObjectName("InfoLabel")
        self.lbl_test_explain.setWordWrap(True)
        self._power_mode_tips = mode_tips
        self._power_test_tips = test_tips
        
        self.p_es = QDoubleSpinBox(); self.p_es.setRange(0.01, 5.0); self.p_es.setValue(0.15); self.p_es.setSingleStep(0.05); self.p_es.setMaximumWidth(120)
        self.p_alp = QDoubleSpinBox(); self.p_alp.setRange(0.001, 0.99); self.p_alp.setValue(0.05); self.p_alp.setDecimals(3); self.p_alp.setMaximumWidth(120)
        self.p_pwr = QDoubleSpinBox(); self.p_pwr.setRange(0.1, 0.99); self.p_pwr.setValue(0.80); self.p_pwr.setMaximumWidth(120)
        self.lbl_es = QLabel("Effect Size:")
        self.lbl_pwr = QLabel("Target Power (1-β):")

        self.p_n_given = QSpinBox(); self.p_n_given.setRange(2, 1_000_000); self.p_n_given.setValue(100); self.p_n_given.setMaximumWidth(120)
        self.lbl_n_given = QLabel("Sample Size You Have (N):")
        self.lbl_n_given.setWordWrap(True)
        self.p_n_given.setVisible(False)
        self.lbl_n_given.setVisible(False)
        
        self.p_groups = QSpinBox(); self.p_groups.setRange(0, 100); self.p_groups.setValue(3) # Changed minimum to 0; self.p_groups.setMaximumWidth(120)
        self.p_groups.setEnabled(False)
        self.lbl_groups = QLabel("Groups/Predictors:")
        self.lbl_groups.setWordWrap(True)
        
        self.p_total_preds = QSpinBox(); self.p_total_preds.setRange(1, 100); self.p_total_preds.setValue(4); self.p_total_preds.setMaximumWidth(120)
        self.lbl_total_preds = QLabel("Total Predictors (incl. interaction):")
        self.lbl_total_preds.setWordWrap(True)
        self.p_total_preds.setVisible(False)
        self.lbl_total_preds.setVisible(False)

        # Two-way ANOVA specific inputs
        self.p_factor_a = QSpinBox(); self.p_factor_a.setRange(2, 50); self.p_factor_a.setValue(2); self.p_factor_a.setMaximumWidth(120)
        self.lbl_factor_a = QLabel("Factor A: Number of Levels:")
        self.p_factor_b = QSpinBox(); self.p_factor_b.setRange(2, 50); self.p_factor_b.setValue(3); self.p_factor_b.setMaximumWidth(120)
        self.lbl_factor_b = QLabel("Factor B: Number of Levels:")
        self.p_twoway_effect = QComboBox(); self.p_twoway_effect.addItems(["Main Effect of Factor A", "Main Effect of Factor B", "Interaction (A × B)"])
        self.lbl_twoway_effect = QLabel("Effect to Power:")
        for w in (self.p_factor_a, self.lbl_factor_a, self.p_factor_b, self.lbl_factor_b, self.p_twoway_effect, self.lbl_twoway_effect):
            w.setVisible(False)

        # Logistic regression specific inputs
        self.p_logistic_p0 = QDoubleSpinBox(); self.p_logistic_p0.setRange(0.01, 0.99); self.p_logistic_p0.setValue(0.50); self.p_logistic_p0.setSingleStep(0.05); self.p_logistic_p0.setMaximumWidth(120)
        self.lbl_logistic_p0 = QLabel("Baseline Event Rate (p at mean of X):")
        self.p_logistic_or = QDoubleSpinBox(); self.p_logistic_or.setRange(1.01, 50.0); self.p_logistic_or.setValue(2.0); self.p_logistic_or.setSingleStep(0.1); self.p_logistic_or.setMaximumWidth(120)
        self.lbl_logistic_or = QLabel("Odds Ratio to Detect:")
        self.p_logistic_r2 = QDoubleSpinBox(); self.p_logistic_r2.setRange(0.0, 0.95); self.p_logistic_r2.setValue(0.0); self.p_logistic_r2.setSingleStep(0.05); self.p_logistic_r2.setMaximumWidth(120)
        self.lbl_logistic_r2 = QLabel("R² of Predictor with Other Predictors:")
        for w in (self.p_logistic_p0, self.lbl_logistic_p0, self.p_logistic_or, self.lbl_logistic_or, self.p_logistic_r2, self.lbl_logistic_r2):
            w.setVisible(False)

        self.lbl_mediation_note = QLabel(
            "Mediation power has no simple closed-form formula — it depends jointly on both the a and b paths "
            "and is best estimated by Monte Carlo simulation, which this calculator does not perform. As a "
            "starting heuristic, bootstrapped mediation analyses are commonly recommended to have N ≥ 100 "
            "(Fritz & MacKinnon, 2007). Please check the power of your specific a and b paths individually "
            "(e.g. as two separate regression/correlation power analyses above) before relying on a single N."
        )
        self.lbl_mediation_note.setWordWrap(True)
        self.lbl_mediation_note.setObjectName("InfoLabel")
        self.lbl_mediation_note.setVisible(False)
        
        def update_power_inputs():
            t = self.p_test.currentText()
            mode = self.p_mode.currentText()
            is_posthoc = mode.startswith("Post-Hoc")
            is_sensitivity = mode.startswith("Sensitivity")
            is_mediation = "Mediation" in t
            is_twoway = "Two-Way" in t
            is_logistic = "Logistic" in t

            mode_idx = self.p_mode.currentIndex()
            test_idx = self.p_test.currentIndex()
            mode_explain = self._power_mode_tips[mode_idx] if 0 <= mode_idx < len(self._power_mode_tips) else ""
            test_explain = self._power_test_tips[test_idx] if 0 <= test_idx < len(self._power_test_tips) else ""
            self.lbl_test_explain.setText(f"<b>{mode.split(':')[0]}:</b> {mode_explain}<br><b>{t}:</b> {test_explain}")

            # Reset visibility every time, then turn on what's relevant
            for w in (self.p_total_preds, self.lbl_total_preds, self.p_factor_a, self.lbl_factor_a,
                      self.p_factor_b, self.lbl_factor_b, self.p_twoway_effect, self.lbl_twoway_effect,
                      self.p_logistic_p0, self.lbl_logistic_p0, self.p_logistic_or, self.lbl_logistic_or,
                      self.p_logistic_r2, self.lbl_logistic_r2, self.lbl_mediation_note,
                      self.p_n_given, self.lbl_n_given, self.lbl_es, self.p_es, self.lbl_pwr, self.p_pwr,
                      self.lbl_groups, self.p_groups):
                w.setVisible(True)

            if is_mediation:
                for w in (self.p_total_preds, self.lbl_total_preds, self.p_factor_a, self.lbl_factor_a,
                          self.p_factor_b, self.lbl_factor_b, self.p_twoway_effect, self.lbl_twoway_effect,
                          self.p_logistic_p0, self.lbl_logistic_p0, self.p_logistic_or, self.lbl_logistic_or,
                          self.p_logistic_r2, self.lbl_logistic_r2, self.p_n_given, self.lbl_n_given,
                          self.lbl_es, self.p_es, self.lbl_pwr, self.p_pwr, self.lbl_groups, self.p_groups):
                    w.setVisible(False)
                self.lbl_mediation_note.setVisible(True)
                return
            else:
                self.lbl_mediation_note.setVisible(False)

            self.p_n_given.setVisible(is_posthoc or is_sensitivity)
            self.lbl_n_given.setVisible(is_posthoc or is_sensitivity)
            # Post-hoc solves for power, so the power input becomes an output (hide it).
            self.lbl_pwr.setVisible(not is_posthoc)
            self.p_pwr.setVisible(not is_posthoc)
            # Sensitivity solves for effect size, so the effect-size input becomes an output (hide it).
            self.lbl_es.setVisible(not is_sensitivity)
            self.p_es.setVisible(not is_sensitivity)

            self.p_twoway_effect.setVisible(is_twoway)
            self.lbl_twoway_effect.setVisible(is_twoway)
            self.p_factor_a.setVisible(is_twoway)
            self.lbl_factor_a.setVisible(is_twoway)
            self.p_factor_b.setVisible(is_twoway)
            self.lbl_factor_b.setVisible(is_twoway)

            self.p_logistic_p0.setVisible(is_logistic)
            self.lbl_logistic_p0.setVisible(is_logistic)
            self.p_logistic_or.setVisible(is_logistic)
            self.lbl_logistic_or.setVisible(is_logistic)
            self.p_logistic_r2.setVisible(is_logistic)
            self.lbl_logistic_r2.setVisible(is_logistic)
            if is_logistic:
                self.lbl_es.setVisible(False)
                self.p_es.setVisible(False)

            if "ANOVA: One-Way" in t:
                self.lbl_groups.setText("Number of Groups (ANOVA):")
                self.p_groups.setEnabled(True)
                self.p_groups.setMinimum(1)
            elif is_twoway:
                self.lbl_groups.setVisible(False)
                self.p_groups.setVisible(False)
            elif "Omnibus R²" in t:
                self.lbl_groups.setText("Total Predictors:")
                self.p_groups.setEnabled(True)
                self.p_groups.setMinimum(1)
            elif "Interaction" in t:
                self.lbl_groups.setText("Tested Predictors (Set to 0 for overall model):")
                self.p_groups.setMinimum(0) # Allow 0 for interaction
                self.p_groups.setEnabled(True)
                self.p_total_preds.setVisible(True)
                self.lbl_total_preds.setVisible(True)
            elif is_logistic:
                self.lbl_groups.setVisible(False)
                self.p_groups.setVisible(False)
            else:
                self.lbl_groups.setText("Groups/Predictors:")
                self.p_groups.setEnabled(False)
            
        self.p_test.currentTextChanged.connect(update_power_inputs)
        self.p_mode.currentTextChanged.connect(update_power_inputs)
        
        form.addRow(QLabel("Mode:"), self.p_mode)
        form.addRow(QLabel("Test Family:"), self.p_test)
        form.addRow(self.lbl_test_explain)

        form.addRow(self.lbl_n_given, self.p_n_given)
        form.addRow(self.lbl_groups, self.p_groups)
        form.addRow(self.lbl_total_preds, self.p_total_preds)

        form.addRow(self.lbl_twoway_effect, self.p_twoway_effect)
        form.addRow(self.lbl_factor_a, self.p_factor_a)
        form.addRow(self.lbl_factor_b, self.p_factor_b)

        form.addRow(self.lbl_logistic_p0, self.p_logistic_p0)
        form.addRow(self.lbl_logistic_or, self.p_logistic_or)
        form.addRow(self.lbl_logistic_r2, self.p_logistic_r2)

        form.addRow(self.lbl_es, self.p_es)
        form.addRow(QLabel("Alpha Error (α):"), self.p_alp)
        form.addRow(self.lbl_pwr, self.p_pwr)

        form.addRow(self.lbl_mediation_note)
        
        btn = QPushButton("▶ Calculate"); btn.clicked.connect(self.run_power)
        btn.setStyleSheet("margin-top: 15px;")
        form.addRow(btn)
        
        cl.addWidget(form_widget)
        cl.addStretch() 
        
        update_power_inputs()
        power_reference_html = (
            "<b>Alpha (α):</b> Probability of a false positive. Standard = .05<br>"
            "<b>Power (1-β):</b> Probability of correctly detecting a real effect. Standard = .80<br><br>"
            "<b>Effect Size Conventions (Cohen, 1988):</b><br>"
            "<i>T-Tests (d):</i> .20 / .50 / .80<br>"
            "<i>ANOVA (f):</i> .10 / .25 / .40<br>"
            "<i>Regression (f²):</i> .02 / .15 / .35<br>"
            "<i>Correlation (r):</i> .10 / .30 / .50<br>"
            "<i>Chi-Square (w):</i> .10 / .30 / .50<br>"
            "<small>(Small / Medium / Large)</small><br><br>"
            "<b>Tip:</b> hover any dropdown option for a when-to-use-this explanation."
        )
        tab, self.power_tabs = self.create_split_module(
            cw, bank=self.build_reference_panel("📏 Effect Size Reference", power_reference_html)
        )
        self.tabs.addWidget(tab)

    def run_power(self):
        try:
            from scipy.optimize import brentq
            t = self.p_test.currentText()
            mode = self.p_mode.currentText()
            a = self.p_alp.value()
            timestamp = datetime.now().strftime("%H:%M:%S")

            # --- Mediation: no closed-form power; show guidance only, no graph ---
            if "Mediation" in t:
                out = self.get_apa_css() + "<h2>Mediation Power</h2>"
                out += (
                    "<div style='background:#FFFBEB; border-left:4px solid #D97706; padding:10px 14px; "
                    "margin:10px 0; font-size:13.5px; color:#78350F;'>"
                    "⚠ <b>This calculator does not compute mediation power.</b> The indirect effect (a×b) "
                    "depends jointly on two regression paths and has no simple closed-form power formula — "
                    "an honest estimate requires Monte Carlo simulation (repeatedly simulating data and running "
                    "the bootstrap mediation test), which is not implemented here. "
                    "As a starting heuristic, bootstrapped mediation analyses are commonly recommended to have "
                    "N ≥ 100 (Fritz &amp; MacKinnon, 2007). Please double-check the power of your specific <i>a</i> "
                    "and <i>b</i> paths individually — run each as its own Correlation or Regression power analysis "
                    "above — rather than relying on a single N for the whole mediation model.</div>"
                )
                tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
                self.power_tabs.addTab(tv, f"Power Calculation ({timestamp})")
                self.power_tabs.setCurrentIndex(self.power_tabs.count() - 1)
                return

            es = self.p_es.value()
            p = self.p_pwr.value()
            n_given = self.p_n_given.value()

            def get_power(N, effect_size, pwr_val_unused=None):
                """Returns achieved power for a given N and effect size. The single
                source of truth that get_size_for_power / get_power_for_n /
                get_min_es_for_n all invert via root-finding, so each test family's
                statistical logic lives in exactly one place."""
                if N is None or N <= 1 or np.isnan(N):
                    return np.nan
                if "Independent" in t:
                    return TTestIndPower().power(effect_size=effect_size, nobs1=N/2, alpha=a, ratio=1.0)
                elif "Paired" in t:
                    return TTestPower().power(effect_size=effect_size, nobs=N, alpha=a)
                elif "ANOVA: One-Way" in t:
                    g = max(1, self.p_groups.value())
                    return FTestAnovaPower().power(effect_size=effect_size, nobs=N, alpha=a, k_groups=g)
                elif "Two-Way" in t:
                    la, lb = max(2, self.p_factor_a.value()), max(2, self.p_factor_b.value())
                    total_cells = la * lb
                    eff = self.p_twoway_effect.currentText()
                    df_num = (la - 1) if "Factor A" in eff else (lb - 1) if "Factor B" in eff else (la - 1) * (lb - 1)
                    df_denom = N - total_cells
                    if df_denom <= 0:
                        return np.nan
                    nc = (effect_size**2) * N
                    fcrit = stats.f.ppf(1 - a, df_num, df_denom)
                    return 1 - stats.ncf.cdf(fcrit, df_num, df_denom, nc)
                elif "Omnibus" in t:
                    u = max(1, self.p_groups.value())
                    v = N - u - 1
                    if v <= 0: return np.nan
                    nc = effect_size * N
                    fcrit = stats.f.ppf(1 - a, u, v)
                    return 1 - stats.ncf.cdf(fcrit, u, v, nc)
                elif "Interaction" in t:
                    u_tested = self.p_groups.value()
                    u_total = self.p_total_preds.value()
                    df_num = u_tested if u_tested > 0 else u_total
                    v = N - u_total - 1
                    if v <= 0: return np.nan
                    nc = effect_size * N
                    fcrit = stats.f.ppf(1 - a, df_num, v)
                    return 1 - stats.ncf.cdf(fcrit, df_num, v, nc)
                elif "Logistic" in t:
                    p0 = self.p_logistic_p0.value()
                    r2_other = self.p_logistic_r2.value()
                    OR = self.p_logistic_or.value()
                    beta = np.log(OR)
                    z_alpha = stats.norm.ppf(1 - a/2)
                    inner = N * p0 * (1 - p0) * (beta**2) * (1 - r2_other)
                    if inner < 0: return np.nan
                    z_stat = np.sqrt(inner) - z_alpha
                    return stats.norm.cdf(z_stat)
                elif "Correlation" in t:
                    return NormalIndPower().power(effect_size=effect_size, nobs1=N, alpha=a)
                elif "Chi-Square" in t:
                    return GofChisquarePower().power(effect_size=effect_size, nobs=N, alpha=a, n_bins=2)
                return np.nan

            def get_size_for_power(pwr_val, effect_size=None):
                """A priori: solve for N given a target power and effect size."""
                es_use = effect_size if effect_size is not None else es
                if "Logistic" in t:
                    p0 = self.p_logistic_p0.value()
                    r2_other = self.p_logistic_r2.value()
                    OR = self.p_logistic_or.value()
                    beta = np.log(OR)
                    z_alpha = stats.norm.ppf(1 - a/2)
                    z_power = stats.norm.ppf(pwr_val)
                    n_req = ((z_alpha + z_power)**2) / (p0 * (1 - p0) * beta**2) / max(1e-9, (1 - r2_other))
                    return np.ceil(n_req)
                if "Independent" in t: return np.ceil(TTestIndPower().solve_power(effect_size=es_use, alpha=a, power=pwr_val, ratio=1.0)) * 2
                elif "Paired" in t: return np.ceil(TTestPower().solve_power(effect_size=es_use, alpha=a, power=pwr_val))
                elif "ANOVA: One-Way" in t:
                    g = max(1, self.p_groups.value())
                    return np.ceil(FTestAnovaPower().solve_power(effect_size=es_use, alpha=a, power=pwr_val, k_groups=g)) * g
                elif "Two-Way" in t:
                    la, lb = max(2, self.p_factor_a.value()), max(2, self.p_factor_b.value())
                    total_cells = la * lb
                    def objective(N):
                        pw = get_power(N, es_use)
                        return (pw if not np.isnan(pw) else -1.0) - pwr_val
                    lo = total_cells + 2
                    if objective(lo) >= 0: return lo
                    try:
                        return np.ceil(brentq(objective, lo, 200000))
                    except Exception:
                        return np.nan
                elif "Omnibus" in t:
                    u = max(1, self.p_groups.value())
                    def objective(N):
                        pw = get_power(N, es_use)
                        return (pw if not np.isnan(pw) else -1.0) - pwr_val
                    if objective(u + 2) >= 0: return u + 2
                    try:
                        return np.ceil(brentq(objective, u + 2, 200000))
                    except Exception:
                        return np.nan
                elif "Interaction" in t:
                    u_total = self.p_total_preds.value()
                    def objective(N):
                        pw = get_power(N, es_use)
                        return (pw if not np.isnan(pw) else -1.0) - pwr_val
                    if objective(u_total + 2) >= 0: return u_total + 2
                    try:
                        return np.ceil(brentq(objective, u_total + 2, 200000))
                    except Exception:
                        return np.nan
                elif "Correlation" in t: return np.ceil(NormalIndPower().solve_power(effect_size=es_use, alpha=a, power=pwr_val))
                elif "Chi-Square" in t: return np.ceil(GofChisquarePower().solve_power(effect_size=es_use, alpha=a, power=pwr_val, n_bins=2))
                return np.nan

            def get_min_es_for_n(N, pwr_val):
                """Sensitivity: solve for the minimum detectable effect size given fixed N and target power."""
                def objective(effect_size):
                    pw = get_power(N, effect_size)
                    return (pw if not np.isnan(pw) else -1.0) - pwr_val
                try:
                    lo, hi = 1e-4, 5.0
                    if objective(hi) < 0:
                        return np.nan  # even a huge effect size can't reach this power at this N
                    return brentq(objective, lo, hi)
                except Exception:
                    return np.nan

            out = self.get_apa_css() + "<h2>Power Analysis Result</h2>"
            graph_x_var = "n"  # what the curve's x-axis sweeps: 'n' or 'es'
            n_for_plot, p_for_plot = None, None

            if mode.startswith("A Priori"):
                n = get_size_for_power(p)
                if pd.isna(n) or np.isnan(n):
                    out += "<p style='font-size:20px; font-weight:bold; color:#EF4444;'>Calculation Failed: Effect size or design is too extreme to solve for N within a reasonable range.</p>"
                else:
                    out += f"<p style='font-size:20px; font-weight:bold; color:#4F46E5;'>Total Required Sample Size (N) = {int(n)}</p>"
                    out += f"<p style='color:#6B7280;'>To detect an effect size of {self.fmt(es)} with {self.fmt(p*100)}% power at α = {self.fmt(a)}.</p>"
                    n_for_plot, p_for_plot = n, p

            elif mode.startswith("Post-Hoc"):
                achieved = get_power(n_given, es)
                if pd.isna(achieved) or np.isnan(achieved):
                    out += "<p style='font-size:20px; font-weight:bold; color:#EF4444;'>Calculation Failed: N is too small for this design (insufficient degrees of freedom).</p>"
                else:
                    color = "#10B981" if achieved >= 0.80 else "#D97706" if achieved >= 0.50 else "#EF4444"
                    out += f"<p style='font-size:20px; font-weight:bold; color:{color};'>Achieved Power = {achieved:.3f} ({achieved*100:.1f}%)</p>"
                    out += f"<p style='color:#6B7280;'>With N = {n_given} and an assumed effect size of {self.fmt(es)} at α = {self.fmt(a)}.</p>"
                    if achieved < 0.80:
                        out += ("<div style='background:#FEF2F2; border-left:4px solid #EF4444; padding:8px 12px; "
                                "margin:10px 0; font-size:13px; color:#7F1D1D;'>⚠ This is below the conventional "
                                "0.80 power threshold (Cohen, 1988) — a true effect of this size had a meaningful "
                                "chance of going undetected (a Type II error) in this study.</div>")
                    n_for_plot, p_for_plot = n_given, achieved

            else:  # Sensitivity
                mde = get_min_es_for_n(n_given, p)
                if pd.isna(mde) or np.isnan(mde):
                    out += "<p style='font-size:20px; font-weight:bold; color:#EF4444;'>Calculation Failed: Could not find a detectable effect size in a reasonable range for this N.</p>"
                else:
                    out += f"<p style='font-size:20px; font-weight:bold; color:#4F46E5;'>Minimum Detectable Effect Size = {mde:.3f}</p>"
                    out += f"<p style='color:#6B7280;'>With N = {n_given} and {self.fmt(p*100)}% power at α = {self.fmt(a)}, this is the smallest true effect you'd reliably detect.</p>"
                    n_for_plot = n_given
                    graph_x_var = "es"
            
            out += "<div class='interpret'><i>Note.</i> <b>Interpretation &amp; Guidelines (Cohen, 1988):</b><br>"
            out += "<b>Alpha (α):</b> The probability of a false positive. Standard cutoff is .05.<br>"
            out += "<b>Power (1-β):</b> The probability of correctly rejecting the null hypothesis. Standard cutoff is .80.<br><br>"
            out += "<b>Effect Size Conventions:</b><br>"
            out += "<i>T-Tests (Cohen's d):</i> Small = 0.20, Medium = 0.50, Large = 0.80<br>"
            out += "<i>ANOVA / Factorial ANOVA (Cohen's f):</i> Small = 0.10, Medium = 0.25, Large = 0.40<br>"
            out += "<i>Multiple Regression (Cohen's f²):</i> Small = 0.02, Medium = 0.15, Large = 0.35<br>"
            out += "<i>Correlation (r):</i> Small = 0.10, Medium = 0.30, Large = 0.50<br>"
            out += "<i>Chi-Square (w):</i> Small = 0.10, Medium = 0.30, Large = 0.50<br>"
            if "Logistic" in t:
                out += "<i>Logistic Regression:</i> Power follows Hsieh, Block &amp; Larsen (1998), Formula 1 for a single continuous predictor, adjusted for multicollinearity via the R² with other predictors.<br>"
            out += "</div>"
            
            tv = QTextEdit(); tv.setReadOnly(True); tv.setHtml(out)
            self.power_tabs.addTab(tv, f"Power Calculation ({timestamp})")
            
            if MATPLOTLIB_AVAILABLE:
                fig = Figure(figsize=(7,5))
                ax = fig.add_subplot(111)
                if self.is_dark_mode:
                    fig.patch.set_facecolor('#1F2937'); ax.set_facecolor('#374151')
                    ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                    ax.title.set_color('white'); ax.tick_params(colors='white')
                
                if graph_x_var == "n":
                    # Power vs. Sample Size curve (a priori and post-hoc modes)
                    powers_range = np.linspace(0.50, 0.99, 25)
                    sizes = [get_size_for_power(p_val) for p_val in powers_range]
                    valid_idx = [i for i, s in enumerate(sizes) if not (s is None or np.isnan(s))]
                    if valid_idx:
                        sizes_clean = [sizes[i] for i in valid_idx]
                        powers_clean = [powers_range[i] for i in valid_idx]
                        ax.plot(sizes_clean, powers_clean, color='#4F46E5', linewidth=2, label="Power Curve")
                    if n_for_plot is not None and not (pd.isna(n_for_plot) or np.isnan(n_for_plot)) and p_for_plot is not None:
                        ax.scatter([n_for_plot], [p_for_plot], color='#EF4444', s=100, zorder=5,
                                   label=f"N={int(n_for_plot)}, Power={p_for_plot:.3f}")
                        ax.axvline(n_for_plot, color='#EF4444', linestyle='--', alpha=0.5)
                        ax.axhline(p_for_plot, color='#EF4444', linestyle='--', alpha=0.5)
                    ax.set_xlabel("Total Sample Size (N)")
                    ax.set_ylabel("Statistical Power (1 - β)")
                else:
                    # Power vs. Effect Size curve (sensitivity mode) -- fixed N, sweep effect size
                    es_range = np.linspace(0.02, 1.5, 40)
                    powers_at_es = [get_power(n_for_plot, es_val) for es_val in es_range]
                    valid_idx = [i for i, pw in enumerate(powers_at_es) if not (pw is None or np.isnan(pw))]
                    if valid_idx:
                        es_clean = [es_range[i] for i in valid_idx]
                        pw_clean = [powers_at_es[i] for i in valid_idx]
                        ax.plot(es_clean, pw_clean, color='#4F46E5', linewidth=2, label=f"Power Curve at N={int(n_for_plot)}")
                    if not (pd.isna(mde) or np.isnan(mde)):
                        ax.scatter([mde], [p], color='#EF4444', s=100, zorder=5, label=f"MDE={mde:.3f}, Power={p:.2f}")
                        ax.axvline(mde, color='#EF4444', linestyle='--', alpha=0.5)
                        ax.axhline(p, color='#EF4444', linestyle='--', alpha=0.5)
                    ax.set_xlabel("Effect Size")
                    ax.set_ylabel("Statistical Power (1 - β)")

                ax.set_title(f"{mode.split(':')[0]} — {t.split(':')[0]}")
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(facecolor='#374151' if self.is_dark_mode else 'white', labelcolor='white' if self.is_dark_mode else 'black')
                
                fig.tight_layout()
                chart = self.make_zoomable_chart(fig)
                self.power_tabs.addTab(chart, f"Power Curve ({timestamp})")
                
            self.power_tabs.setCurrentIndex(self.power_tabs.count() - 1)
        except Exception as e:
            self.power_tabs.addTab(QTextEdit(str(e)), "Error")

# ==========================================
# APPLICATION ENTRY POINT
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Optional: Apply a modern font globally
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = PsyStat()
    window.show()
    sys.exit(app.exec())
