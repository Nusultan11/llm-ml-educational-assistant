# Contributing

## Quick start

1. Create a virtual environment.
2. Install dependencies.
3. Install package in editable mode.
4. Run tests before opening a PR.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

## Coding rules

- Keep production logic in `src/`.
- Keep experiments in `notebooks/`.
- Add or update tests in `tests/` for any behavior change.
- Keep configuration in `configs/`.

## Pull request checklist

- [ ] Tests pass locally.
- [ ] README updated if behavior changed.
- [ ] No large binaries or private data committed.
- [ ] Notebook outputs minimized.
