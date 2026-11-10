# Contributing to PsyStat

Thank you for helping to improve PsyStat! Bug reports, corrections to statistical output, documentation fixes, new
examples and code contributions are all welcome.

## Ways to contribute

* **Report a bug** or a **result that disagrees with another program** (R, SPSS, JASP...): open an issue using the templates.
  Statistical discrepancies are especially valuable; include the data (anonymised or simulated), the settings, PsyStat's
  output and the reference output.
* **Suggest a feature**, a method or a reference that should be added.
* **Improve the documentation** (`docs/USER_MANUAL.md`, `examples/EXAMPLES_GUIDE.md`).
* **Contribute code** through a pull request.

**Never post real participant data.** Use the simulated example datasets or fully anonymised data.

## Development setup

```bash
git clone https://github.com/<your-username>/psystat.git
cd psystat
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python psystat.py                                       # run the app
pytest                                                  # run the tests
python psystat.py --selftest                            # verify that all libraries load
```

Python 3.11 or 3.12 is recommended. Tests run headless (`QT_QPA_PLATFORM=offscreen` is set automatically).

## Project layout

| Path | Contents |
|---|---|
| `psystat.py` | The application (GUI and analyses) |
| `resources.py` | Locating bundled files; example data; opening documents (works in frozen builds) |
| `docs/` | User manual (Markdown source, HTML and PDF built from it) |
| `examples/` | Simulated example datasets and the guide |
| `tools/` | `generate_examples.py`, `build_docs.py`, `make_icons.py` |
| `tests/` | Automated tests |
| `psystat.spec`, `installer/` | Packaging for Windows and macOS |
| `.github/` | Issue templates and GitHub Actions workflows |

## Pull requests

1. Open an issue first for anything large, so we can agree on the approach.
2. Create a branch from `main`; keep each pull request focused on one topic.
3. **Statistical changes need a reference:** cite the method (paper or textbook) in the code comment and in the manual,
   and add a test that compares the result with a known value or an independent implementation.
4. Run `pytest` and `python psystat.py --selftest` before submitting.
5. If you change something that users see, update `docs/USER_MANUAL.md` and `CHANGELOG.md`
   (and rebuild the HTML/PDF with `python tools/build_docs.py`).
6. Be careful with lazily imported libraries: every package the app imports on demand must be listed in `psystat.spec`
   (`hiddenimports`) and in `requirements.txt`.

## Style

* Follow the existing structure and naming; prefer clear code to clever code.
* Output text should follow APA 7 conventions (italic statistical symbols, no leading zero for *p*, explicit notes).
* Never swallow an error silently in an analysis: either handle it with a clear message to the user or let it surface.

## Code of Conduct

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

By contributing you agree that your contributions are licensed under the project's [MIT License](LICENSE).
