"""
resources.py - locate bundled files and open documentation.

Works when PsyStat is run from source AND when it is frozen with PyInstaller on Windows or macOS
(onedir / .app bundles).  Public API used by psystat.py:

    resource_path(relative_path)        -> absolute path of a bundled file (raises FileNotFoundError)
    load_example_dataset(filename)      -> pandas.DataFrame
    get_example_dataset_info()          -> {filename: {"label": ..., "description": ...}}
    open_user_manual()                  -> opens docs/USER_MANUAL.html (or renders the .md) in the browser
    open_markdown_document(rel, title)  -> renders a bundled .md file to HTML and opens it in the browser
    render_markdown(md_text, title)     -> full HTML page (string)
"""
import html as _html
import os
import pathlib
import sys
import tempfile
import webbrowser

# ─────────────────────────────────────────────────────────────────────────────
# Locating bundled files
# ─────────────────────────────────────────────────────────────────────────────
def _candidate_bases():
    """Folders that may contain the bundled data files, most specific first."""
    bases = []
    meipass = getattr(sys, "_MEIPASS", None)               # PyInstaller (onefile temp dir / onedir _internal)
    if meipass:
        bases.append(meipass)
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    if getattr(sys, "frozen", False):
        bases += [
            exe_dir,
            os.path.join(exe_dir, "_internal"),                         # PyInstaller >= 6 onedir (Windows)
            os.path.abspath(os.path.join(exe_dir, "..", "Resources")),  # macOS .app bundle
            os.path.abspath(os.path.join(exe_dir, "..", "Frameworks")),
        ]
    bases.append(os.path.dirname(os.path.abspath(__file__)))           # running from source
    bases.append(os.path.abspath("."))                                 # last resort: current folder
    seen, out = set(), []
    for b in bases:
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return out


def resource_path(relative_path):
    """Absolute path of a bundled file or folder. Raises FileNotFoundError if it cannot be found."""
    rel = str(relative_path).replace("\\", "/")
    for base in _candidate_bases():
        p = os.path.join(base, *rel.split("/"))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Bundled file not found: '{relative_path}'.\nSearched in:\n  " + "\n  ".join(_candidate_bases())
    )


# ─────────────────────────────────────────────────────────────────────────────
# Example datasets
# ─────────────────────────────────────────────────────────────────────────────
_EXAMPLE_DATASETS = {
    "experiment_example.csv": {
        "label": "Experiment Example (N=300) — Contact Study",
        "description": ("Simulated two-group (Contact vs Control) experiment with pre-test, post-test and follow-up "
                        "measures of prejudice, contact quality, group identification, intergroup anxiety, cooperation "
                        "and life satisfaction. Includes mediation, moderation and mixed-design effects, and some missing data."),
    },
    "survey_example.csv": {
        "label": "Survey Example (N=250) — Psychometric Survey",
        "description": ("Simulated cross-sectional survey: 15 five-point Likert items measuring four constructs "
                        "(Social Identity, Intergroup Contact, Prejudice, Wellbeing) plus age, gender and education. "
                        "Item si5 is reverse-worded. About 1% of item responses are missing."),
    },
    "cvi_expert_ratings_example.csv": {
        "label": "CVI Example — Expert Ratings (12 items x 6 experts)",
        "description": ("Simulated relevance ratings (1-4) of 12 items by 6 experts, for the Content Validity Index "
                        "(I-CVI, S-CVI/Ave, S-CVI/UA) calculator in Item Analysis & CVI."),
    },
    "timeseries_example.csv": {
        "label": "Time Series Example — Monthly Counselling Visits (60 months)",
        "description": ("Simulated monthly counselling-centre visits with an upward trend and yearly seasonality, "
                        "for Forecasting (Time Series)."),
    },
}


def get_example_dataset_info():
    return {k: dict(v) for k, v in _EXAMPLE_DATASETS.items()}


def load_example_dataset(filename):
    import pandas as pd
    return pd.read_csv(resource_path("examples/" + os.path.basename(filename)))


# ─────────────────────────────────────────────────────────────────────────────
# Markdown -> HTML (used for the README / examples guide, and as a fallback for the manual)
# ─────────────────────────────────────────────────────────────────────────────
HTML_CSS = """
body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; color:#1f2933; line-height:1.55;
       max-width: 940px; margin: 0 auto; padding: 24px 28px 60px 28px; font-size: 15px; }
h1 { font-size: 30px; color:#14213d; border-bottom: 3px solid #14213d; padding-bottom: 8px; margin-top: 8px; }
h2 { font-size: 23px; color:#14213d; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; margin-top: 38px; }
h3 { font-size: 18px; color:#1d3557; margin-top: 26px; }
h4 { font-size: 15px; color:#1d3557; margin-top: 18px; }
a { color:#1d4ed8; text-decoration: none; } a:hover { text-decoration: underline; }
code { background:#eef2f7; padding: 1px 5px; border-radius: 3px; font-family: Consolas, 'Courier New', monospace; font-size: 13px; }
pre { background:#0f172a; color:#e2e8f0; padding: 12px 14px; border-radius: 6px; overflow-x: auto; }
pre code { background: transparent; color: inherit; padding: 0; }
table { border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 14px; }
th, td { border: 1px solid #cbd5e1; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background:#e8eef6; }
tr:nth-child(even) td { background:#f8fafc; }
blockquote { border-left: 4px solid #94a3b8; margin: 14px 0; padding: 4px 14px; background:#f1f5f9; color:#334155; }
hr { border: none; border-top: 1px solid #cbd5e1; margin: 28px 0; }
img { max-width: 100%; }
.toc { background:#f8fafc; border:1px solid #e2e8f0; border-radius: 6px; padding: 8px 18px; }
@media print { body { max-width: none; padding: 0; } h2 { page-break-before: auto; } pre, table, blockquote { page-break-inside: avoid; } }
"""


def _md_to_body(md_text):
    """Markdown text -> HTML body. Tries mistune, then python-markdown, then a plain <pre> fallback."""
    try:
        import mistune
        try:
            md = mistune.create_markdown(escape=False, plugins=["table", "strikethrough", "task_lists"])
        except Exception:
            md = mistune.create_markdown(escape=False, plugins=["table", "strikethrough"])
        return md(md_text)
    except Exception:
        pass
    try:
        import markdown
        return markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    except Exception:
        return "<pre>" + _html.escape(md_text) + "</pre>"


def render_markdown(md_text, title="PsyStat"):
    body = _md_to_body(md_text)
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>{_html.escape(title)}</title>"
            f"<style>{HTML_CSS}</style></head><body>{body}</body></html>")


def _open_in_browser(path):
    """Open a local file in the default browser (valid file:// URI on Windows and macOS)."""
    return webbrowser.open(pathlib.Path(path).resolve().as_uri())


def open_markdown_document(relative_path, title=None):
    """Render a bundled Markdown file to HTML and open it in the default browser."""
    src = resource_path(relative_path)
    with open(src, "r", encoding="utf-8") as fh:
        text = fh.read()
    page = render_markdown(text, title or os.path.basename(src))
    tmp = tempfile.NamedTemporaryFile(prefix="psystat_doc_", suffix=".html", delete=False, mode="w", encoding="utf-8")
    tmp.write(page)
    tmp.close()
    return _open_in_browser(tmp.name)


def open_user_manual():
    """Open the user manual in the browser: the pre-rendered HTML if bundled, otherwise render the Markdown."""
    try:
        return _open_in_browser(resource_path("docs/USER_MANUAL.html"))
    except FileNotFoundError:
        return open_markdown_document("docs/USER_MANUAL.md", "PsyStat User Manual")
