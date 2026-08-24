# CLAUDE.md

This is a research repo for an event study on post-earnings announcement drift.

- Analysis logic lives in `src/pead/`; notebooks import from it rather than defining logic inline.
- All modelling decisions are recorded in `docs/methodology.md`, which must be read before changing any modelling code.
- Never commit data files.
- Prefer statsmodels over scikit-learn — this project needs statistical inference, not prediction.
- Any function doing arithmetic on returns needs a pytest test.
- The virtual environment is at `.venv`; dependencies are in `requirements.txt`, not `pyproject.toml`.
