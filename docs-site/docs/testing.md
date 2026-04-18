# Testing

Run full test suite:

```bash
python -m pytest -q
```

## Coverage Focus

- mention extraction and normalization
- week bucketing and year-boundary behavior
- weekly graph node/edge aggregation
- integration artifact generation
- global animation replay and deterministic layout
- app helper smoke checks

## Recommended Local Checks

```bash
python -m compileall src app
python -m pytest -q
```

## Typical Issue on Windows

If pytest temp directory permissions fail, run:

```bash
python -m pytest -q --basetemp=.pytest_tmp
```
