# Contributing to PsyStat

Thank you for your interest in PsyStat. Contributions of all kinds are welcome —
bug reports, feature suggestions, documentation improvements, and code contributions.

---

## Reporting a Bug

Before opening an issue, please check the existing
[Issues](../../issues) to see if the problem has already been reported.

When opening a new issue, include:

1. **PsyStat version** — visible in the app title bar or About dialog
2. **Operating system** — Windows version or macOS version
3. **What you did** — the steps you took before the error occurred
4. **What you expected** — what you thought should happen
5. **What actually happened** — the error message or incorrect output
6. **Sample data** — if possible, a small anonymised CSV that reproduces
   the problem (do not share real participant data)

Use the **Bug Report** issue template when available.

---

## Suggesting a Feature

Open an issue using the **Feature Request** template. Describe:

- What you are trying to do statistically
- Why the current functionality does not cover it
- Which software (SPSS, R, Mplus, JASP, jamovi) covers it today, if any
- Any references to the method you would like added

---

## Contributing Code

1. **Fork** the repository and create a new branch from `main`:
   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes in `psystat.py`. Please follow the existing code style:
   - Functions are methods of the main `PsyStat` class
   - UI init functions are named `init_<module>_tab()`
   - Run functions are named `run_<module>()`
   - HTML output uses the `get_apa_css()` stylesheet and `self.fmt()` for numbers

3. Test your changes against both example datasets
   (`examples/experiment_example.csv` and `examples/survey_example.csv`).

4. If your change affects statistical output, document the method and any
   known differences from SPSS in a comment block above the function.

5. Open a **Pull Request** against `main` with a clear description of what
   changed and why.

---

## Statistical Accuracy

PsyStat aims for high parity with SPSS output. If you are implementing or
modifying a statistical procedure, please:

- Cite the primary reference for the algorithm
- Note which SPSS procedure the output is intended to match
- Document any expected numerical differences (e.g., optimiser tolerance
  differences in EFA) in the output itself and in the User Manual

---

## Code of Conduct

Be respectful, constructive, and collegial. This project is maintained by
an academic researcher; please be patient with response times.
