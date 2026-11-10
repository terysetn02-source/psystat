## What does this change?

<!-- A short description and the issue it addresses (Fixes #...) -->

## Type of change

- [ ] Bug fix
- [ ] New analysis / option
- [ ] Change to statistical output
- [ ] Documentation or examples
- [ ] Packaging / CI

## Checklist

- [ ] `pytest` passes and `python psystat.py --selftest` reports ALL CHECKS PASSED
- [ ] Statistical changes cite their method (paper or textbook) and include a test against a known value or an independent implementation
- [ ] `docs/USER_MANUAL.md` and `CHANGELOG.md` are updated if users can see the change (HTML/PDF rebuilt with `python tools/build_docs.py`)
- [ ] Any newly imported library is listed in `requirements.txt` and in `psystat.spec`
- [ ] No real participant data is included
