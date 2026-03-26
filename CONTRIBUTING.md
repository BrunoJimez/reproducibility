# Contributing to Reproducibility

Thank you for your interest in contributing. This document describes how to
report bugs, propose improvements, and submit code changes.

---

## Reporting bugs

Open an issue at https://github.com/BrunoJimez/reproducibility/issues.

Please include:
- **Python version** and OS
- **Full error message** (paste the complete traceback)
- **Minimal reproducible example** — the shortest code that triggers the bug
- **Expected vs. actual behaviour**

---

## Proposing new features

Open an issue labelled `enhancement` describing:
- The scientific use case that motivates the feature
- How it relates to the existing composite score / AIC framework
- Whether it requires new dependencies (new dependencies must be open-source
  and free; proprietary or paid APIs are not accepted)

---

## Submitting code changes

### Setup

```cmd
git clone https://github.com/BrunoJimez/reproducibility.git
cd reproducibility
python -m venv .venv && .venv\Scripts\activate   :: Windows
pip install -r requirements.txt && pip install pytest
```

### Quality requirements

All pull requests must satisfy:

```cmd
pytest tests\ -v
```

**Result must be: 52 passed** (zero failures, warnings are acceptable).

The `TestScoreValidation` class verifies the scientific validity of the
scoring instrument. Any change to `reproducibility_core.py` that causes
these six tests to fail will not be merged, regardless of other merits.

### Code standards

- All code, comments, docstrings, and user-facing strings must be in **English**
- New modules must have at least 4 tests covering normal operation and edge cases
- Functions longer than 30 lines should have a docstring explaining inputs,
  outputs, and any non-obvious design decisions
- No new proprietary or paid dependencies

### Pull request checklist

```
[ ] pytest tests\ -v  ->  52 passed
[ ] New code has docstrings
[ ] User-facing strings are in English
[ ] No new paid/proprietary dependencies
[ ] REPRODUCTION_GUIDE.md updated if CLI commands or output formats changed
[ ] paper.md updated if the scientific behaviour of any module changed
```

---

## Scientific contribution standards

This software makes quantitative scientific claims (reproducibility scores
validated against published literature). Contributions that modify the
scoring formulae, validation cases, or NLP translation layers require:

1. A scientific justification (reference to peer-reviewed literature preferred)
2. At least one new test in `TestScoreValidation` or equivalent
3. An update to the paper.md Theoretical Background or Limitations section

---

## Licence

By contributing, you agree that your contributions will be released under
the project's MIT License.
