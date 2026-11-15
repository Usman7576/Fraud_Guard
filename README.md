# FraudGuard

Cost-aware fraud detection for online card payments: a scoring API that chooses its decision threshold by minimising business cost, explains every decision, and detects when it needs retraining. It comes with an analyst console and a 3D landing page.

Built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (590,540 e-commerce transactions, 3.5% fraud). It is evaluated on future data, in money terms, with confidence intervals.

> **Status:** week 1 of 10, the data foundation. The full plan is in [docs/FraudGuard_Project_Plan.pdf](docs/FraudGuard_Project_Plan.pdf).

## Stack

| Layer | Tools |
|---|---|
| ML and data | Python 3.12, Polars, LightGBM, Optuna, SHAP, MLflow, Evidently |
| Backend | FastAPI, Pydantic v2, MongoDB |
| Frontend | Next.js, React, TypeScript, Tailwind, shadcn/ui, React Three Fiber |
| Tooling | uv, Typer CLI, ruff, mypy, pytest, Hypothesis, GitHub Actions |

Everything runs locally. There is no Docker and no GPU.

## Quick start

```bash
uv sync                               # installs Python 3.12 deps from uv.lock
uv run fraudguard data synth          # optional: synthetic data, until the Kaggle files are in place
uv run fraudguard all                 # rebuild every stage
uv run fraudguard data info           # show the current snapshot manifest
```

### Getting the real data

1. Accept the competition rules at https://www.kaggle.com/c/ieee-fraud-detection.
2. Download `train_transaction.csv` and `train_identity.csv` into `data/raw/`.
3. Run `uv run fraudguard data build --force`.

## Data foundation (week 1)

- **Time split.** Train on Dec–Mar, validate on Apr, test on May. The boundaries are in [configs/data.yaml](configs/data.yaml). A Hypothesis property test checks that no split reaches into a later one, whatever the boundaries.
- **Card identity.** IEEE-CIS has no card ID. `card_uid` combines `card1`, `addr1` and the account start day (`day - D1`). This is an approximation, and its label consistency is reported on every build.
- **Versioned snapshots.** Each stage writes Parquet plus a `manifest.json` with SHA-256 hashes, row counts, schema, input hashes, a config fingerprint and the code version. A stage is skipped when its inputs and config are unchanged. Tampered files are rejected.

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check (strict)
uv run pre-commit install
```
