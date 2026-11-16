# BankGuard

![CI](https://github.com/sap6011/bank-guard/actions/workflows/ci.yml/badge.svg)

Money laundering detection on the IBM AML HI-Small dataset, with a multi-agent investigation workflow (in progress).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Place the HI-Small files in `data/`.

```
uv sync
uv run pytest
uv run python -m bankguard.train
```

The last command trains the current model and prints validation metrics (PR-AUC 0.3412).