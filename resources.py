"""
resources.py — PsyStat built-in resource loader
------------------------------------------------
Resolves paths to bundled files (User Manual, example datasets) whether the
app is running:
  A. From source          : /path/to/repo/psystat.py
  B. As a frozen bundle   : PyInstaller / Nuitka .exe or .app

Usage inside psystat.py
-----------------------
    from resources import resource_path, open_user_manual, load_example_dataset

    # Get an absolute path to any bundled file
    manual_path   = resource_path("docs/USER_MANUAL.md")
    csv_path      = resource_path("examples/survey_example.csv")

    # Open the User Manual in the system's default browser/Markdown viewer
    open_user_manual()

    # Load an example dataset directly into a pandas DataFrame
    df = load_example_dataset("experiment_example.csv")
    df = load_example_dataset("survey_example.csv")
"""

import os
import sys
import webbrowser
import tempfile
import shutil

import pandas as pd


# ── Path resolution ───────────────────────────────────────────────────────────

def _base_dir() -> str:
    """
    Return the base directory that contains docs/, examples/, etc.

    When frozen by PyInstaller:  sys._MEIPASS  (the temp extraction folder)
    When frozen by Nuitka:       directory of the executable
    When running from source:    directory of this file (= repo root)
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller sets sys._MEIPASS; Nuitka does not, but sets sys.frozen
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(relative_path: str) -> str:
    """
    Return the absolute path to a bundled resource file.

    Parameters
    ----------
    relative_path : str
        Path relative to the repo/bundle root, using forward slashes.
        Examples: "docs/USER_MANUAL.md", "examples/survey_example.csv"

    Returns
    -------
    str
        Absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    path = os.path.join(_base_dir(), relative_path.replace("/", os.sep))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Bundled resource not found: {path}\n"
            f"Base directory resolved to: {_base_dir()}\n"
            "If running from source, ensure docs/ and examples/ are present "
            "next to psystat.py."
        )
    return path


# ── User Manual ───────────────────────────────────────────────────────────────

def open_user_manual() -> None:
    """
    Open the built-in User Manual in the system's default browser.

    The Markdown file is converted to a minimal HTML page and written to a
    temporary file, then opened via webbrowser.open().  The HTML renders
    correctly in any modern browser without any extra tooling.
    """
    try:
        md_path = resource_path("docs/USER_MANUAL.md")
    except FileNotFoundError as exc:
        _show_error(str(exc))
        return

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html = _md_to_html(md_text)

    # Write to a temp file that persists until the OS cleans it up
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()
    webbrowser.open(f"file://{tmp.name}")


def _md_to_html(md: str) -> str:
    """
    Convert Markdown to a self-contained HTML page.

    Handles: headings, bold/italic, inline code, fenced code blocks,
    horizontal rules, tables (GitHub-Flavored), blockquotes, and lists.
    External CSS-free; uses inline styles compatible with all browsers.
    """
    # Try mistune first (fast, no external deps other than itself)
    try:
        import mistune
        body = mistune.html(md)
    except ImportError:
        # Minimal regex-based fallback — covers the constructs in USER_MANUAL.md
        import re

        lines = md.split("\n")
        out   = []
        in_code  = False
        in_table = False
        code_buf = []

        for line in lines:
            # Fenced code blocks
            if line.startswith("```"):
                if in_code:
                    out.append("<pre><code>" + "\n".join(code_buf) + "</code></pre>")
                    code_buf = []
                    in_code  = False
                else:
                    in_code = True
                continue
            if in_code:
                code_buf.append(_escape(line))
                continue

            # Tables
            if "|" in line and line.strip().startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if all(set(c.replace("-", "").replace(":", "")) == set() for c in cells):
                    continue   # separator row
                if not in_table:
                    out.append("<table style='border-collapse:collapse;width:100%;margin:12px 0'>")
                    tag = "th"
                    in_table = True
                else:
                    tag = "td"
                row = "".join(
                    f"<{tag} style='border:1px solid #ccc;padding:6px 10px;text-align:left'>"
                    f"{_inline(c)}</{tag}>"
                    for c in cells
                )
                out.append(f"<tr>{row}</tr>")
                continue
            elif in_table:
                out.append("</table>")
                in_table = False

            # Headings
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if m:
                lvl  = len(m.group(1))
                text = _inline(m.group(2))
                size = {1: "2em", 2: "1.5em", 3: "1.2em"}.get(lvl, "1em")
                out.append(f"<h{lvl} style='font-size:{size};margin-top:1.2em'>{text}</h{lvl}>")
                continue

            # Horizontal rule
            if re.match(r"^---+$", line.strip()):
                out.append("<hr style='border:none;border-top:1px solid #ddd;margin:16px 0'>")
                continue

            # Blockquote
            if line.startswith("> "):
                out.append(
                    f"<blockquote style='border-left:4px solid #ccc;margin:8px 0;"
                    f"padding:4px 12px;color:#555'>{_inline(line[2:])}</blockquote>"
                )
                continue

            # Unordered list
            if re.match(r"^[-*]\s", line):
                out.append(f"<li>{_inline(line[2:])}</li>")
                continue

            # Ordered list
            if re.match(r"^\d+\.\s", line):
                cleaned_line = re.sub(r'^\d+\.\s', '', line)
                out.append(f"<li>{_inline(cleaned_line)}</li>")
                continue

            # Blank line
            if line.strip() == "":
                out.append("<br>")
                continue

            out.append(f"<p style='margin:6px 0'>{_inline(line)}</p>")

        if in_table:
            out.append("</table>")
        if in_code:
            out.append("<pre><code>" + "\n".join(code_buf) + "</code></pre>")

        body = "\n".join(out)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PsyStat User Manual</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 900px; margin: 40px auto; padding: 0 24px;
          color: #1a1a1a; line-height: 1.6; }}
  code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px;
          font-family: 'Courier New', monospace; font-size: .9em; }}
  pre  {{ background: #f4f4f4; padding: 14px; border-radius: 6px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 7px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  blockquote {{ border-left: 4px solid #bbb; margin: 10px 0;
                padding: 6px 14px; color: #555; background: #fafafa; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 2em; }}
  a  {{ color: #2563EB; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(text: str) -> str:
    """Apply inline Markdown formatting (bold, italic, code, links)."""
    import re
    text = _escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    text = re.sub(r"`(.+?)`",       r"<code>\1</code>",      text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return text


def _show_error(msg: str) -> None:
    """Display an error if the manual cannot be opened."""
    try:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(None, "PsyStat — Resource Error", msg)
    except Exception:
        print(f"[PsyStat resource error] {msg}")


# ── Example datasets ──────────────────────────────────────────────────────────

#: Available built-in example datasets
EXAMPLE_DATASETS = {
    "experiment_example.csv": {
        "label": "Intergroup Contact Experiment (N=300)",
        "description": (
            "Experimental study with pre/post measures of prejudice, contact quality, "
            "group identification, intergroup anxiety, cooperation, and life satisfaction. "
            "Suitable for: Paired T-Tests, Repeated-Measures ANOVA, Regression, "
            "Mediation Analysis, SEM."
        ),
        "path": "examples/experiment_example.csv",
    },
    "survey_example.csv": {
        "label": "Cross-Sectional Survey Study (N=250)",
        "description": (
            "Survey with 15 Likert-scale items across four constructs: Social Identity (5 items), "
            "Intergroup Contact (4 items), Prejudice (3 items), and Wellbeing (3 items). "
            "Suitable for: Item Analysis, EFA, CFA, Correlation, CATPCA, Network Analysis, SEM."
        ),
        "path": "examples/survey_example.csv",
    },
}


def load_example_dataset(filename: str) -> pd.DataFrame:
    """
    Load a built-in example dataset and return it as a pandas DataFrame.

    Parameters
    ----------
    filename : str
        One of: "experiment_example.csv", "survey_example.csv"

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the filename is not a recognised built-in dataset.
    FileNotFoundError
        If the file cannot be found inside the bundle.
    """
    if filename not in EXAMPLE_DATASETS:
        raise ValueError(
            f"Unknown example dataset: {filename!r}. "
            f"Available: {list(EXAMPLE_DATASETS.keys())}"
        )
    path = resource_path(EXAMPLE_DATASETS[filename]["path"])
    return pd.read_csv(path)


def get_example_dataset_info() -> dict:
    """Return metadata dict for all built-in example datasets."""
    return EXAMPLE_DATASETS
