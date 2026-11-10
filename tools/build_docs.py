"""Build docs/USER_MANUAL.html and docs/USER_MANUAL.pdf from docs/USER_MANUAL.md.

Requirements (development only):
    pip install markdown pdfkit
    plus the wkhtmltopdf program for the PDF  (https://wkhtmltopdf.org/downloads.html)

Usage:
    python tools/build_docs.py            # HTML + PDF (PDF skipped with a message if wkhtmltopdf is missing)
    python tools/build_docs.py --html     # HTML only
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import markdown                                   # noqa: E402
from resources import HTML_CSS                    # noqa: E402  (same styling as the in-app documents)

SRC = os.path.join(ROOT, "docs", "USER_MANUAL.md")
OUT_HTML = os.path.join(ROOT, "docs", "USER_MANUAL.html")
OUT_PDF = os.path.join(ROOT, "docs", "USER_MANUAL.pdf")


def gh_slug(value, separator="-"):
    """GitHub-style heading anchors, so the same '#8-item-analysis--cvi' links work on GitHub and in the HTML."""
    value = value.strip().lower()
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    return value.replace(" ", separator)


def version():
    try:
        with open(os.path.join(ROOT, "psystat.py"), encoding="utf-8") as fh:
            m = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', fh.read())
        return m.group(1) if m else "?"
    except OSError:
        return "?"


def render_html():
    with open(SRC, encoding="utf-8") as fh:
        text = fh.read()
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        extension_configs={"toc": {"slugify": gh_slug}},
    )
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>PsyStat User Manual (v{version()})</title><style>{HTML_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote", OUT_HTML)
    return page


def render_pdf(page):
    try:
        import pdfkit
        options = {
            "page-size": "A4", "margin-top": "18mm", "margin-bottom": "18mm",
            "margin-left": "16mm", "margin-right": "16mm", "encoding": "UTF-8",
            "footer-center": "PsyStat User Manual   |   page [page] of [topage]", "footer-font-size": "8",
            "enable-local-file-access": None, "quiet": "",
        }
        pdfkit.from_string(page, OUT_PDF, options=options)
        print("wrote", OUT_PDF)
    except Exception as exc:                                   # wkhtmltopdf missing or failed
        print("PDF not built:", exc)
        print("Install wkhtmltopdf, or print docs/USER_MANUAL.html to PDF from your browser.")


if __name__ == "__main__":
    html_page = render_html()
    if "--html" not in sys.argv:
        render_pdf(html_page)
