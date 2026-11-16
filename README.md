# FraudGuard

Cost-aware fraud detection for online card payments: a scoring API that chooses its decision threshold by minimising business cost, explains every decision, and detects when it needs retraining. It comes with an analyst console and a 3D landing page.

Built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (590,540 e-commerce transactions, 3.5% fraud). It is evaluated on future data, in money terms, with confidence intervals.

> **Status:** week 4 of 10, tuned LightGBM, calibration and the cost-based decision policy. The full plan is in [docs/FraudGuard_Project_Plan.pdf](docs/FraudGuard_Project_Plan.pdf).

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
uv run fraudguard features build      # build the feature snapshot
uv run fraudguard report adversarial  # train-vs-test drift report
uv run fraudguard train baselines     # rules and logistic regression
uv run fraudguard train lightgbm      # tune, calibrate, choose the policy, register the champion
```

### Getting the real data

1. Accept the competition rules at https://www.kaggle.com/c/ieee-fraud-detection.
2. Download `train_transaction.csv` and `train_identity.csv` into `data/raw/`.
3. Run `uv run fraudguard data build --force`.

## Data foundation (week 1)

- **Time split.** Train on Dec–Mar, validate on Apr, test on May. The boundaries are in [configs/data.yaml](configs/data.yaml). A Hypothesis property test checks that no split reaches into a later one, whatever the boundaries.
- **Card identity.** IEEE-CIS has no card ID. `card_uid` combines `card1`, `addr1` and the account start day (`day - D1`). This is an approximation, and its label consistency is reported on every build.
- **Versioned snapshots.** Each stage writes Parquet plus a `manifest.json` with SHA-256 hashes, row counts, schema, input hashes, a config fingerprint and the code version. A stage is skipped when its inputs and config are unchanged. Tampered files are rejected.

## Features (week 2)

Every aggregate uses only transactions with an **earlier timestamp** than the current one. Rows in the same second are excluded, so the online scorer can reproduce each value exactly. Windows and keys are in [configs/features.yaml](configs/features.yaml).

| Group | Columns | What it captures |
|---|---|---|
| Base | raw C, D, M, V and id columns, card, address, email and device fields, plus `amt_log`, `amt_cents`, `hour`, `weekday` | Per-row facts; cannot leak |
| Velocity | count and amount per card, email domain and device over 1h, 24h and 7d (18 columns) | Bursts of activity |
| Behaviour | prior transaction count, new-card flag, seconds since last, amount z-score and ratio, hour-of-day deviation (circular mean) | How unusual this is *for this card* |

**Leakage test.** [`features/leakage.py`](src/fraudguard/features/leakage.py) cuts the data at five points in time and rebuilds every feature without the future rows. If any past value changes, the test fails and names the column. To prove the test can catch leaks, [tests/test_leakage.py](tests/test_leakage.py) also runs it on five deliberately leaky features: a group-wide mean, a forward shift, label target encoding, a window that looks ahead, and global normalisation. The test must flag every one.

**Adversarial validation.** `fraudguard report adversarial` trains LightGBM to tell training rows from test rows (5-fold CV). It reports the AUC and the features that drift most to `reports/adversarial_validation.{json,md}`. Lifetime counters such as `card_prior_txn` and `D1` grow with time, so they separate later months by construction. That is worth knowing before modelling, and the ablation study will show whether they earn their place.

## Entity graph and baselines (week 3)

**Entity graph.** [`features/graph.py`](src/fraudguard/features/graph.py) links cards to devices, billing addresses and email domains. It streams transactions in time order and scores each one against the graph as it stood before that second. The 10 graph columns cover:
- devices, addresses and emails per card
- distinct cards per device and per address over the last 30 days
- prior fraud known on this card
- known-fraud cards on the same device or address
- cards and known-fraud cards within **two hops** (card → device/address → card)

Labels respect a **30-day chargeback delay**: a fraudulent transaction only marks its card as fraud once 30 days have passed. Email domains are not used for hops, because gmail.com would link almost everyone. Entities with more than 200 cards are not expanded. The `EntityGraph` class is shared with the online scorer, so live features match training.

**What adversarial validation caught.** The first version counted all cards ever seen on an email domain, which acts as a clock. It pushed adversarial AUC to 1.0 and was 67% of the drift signal. Entity sizes are now rolling 30-day counts and the email count is gone. Other all-time counters (blacklist hits, lifetime transactions, 2-hop neighbours) still drift by nature. Whether they earn their place despite the drift is a question for the ablation study on real data.

**Baselines** (`fraudguard train baselines`). Both baselines are fitted on train and compared on the validation month. Every run is tracked in MLflow (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`) with the data and feature snapshot digests.

| Model | What it is |
|---|---|
| Rules | Six hand-written rules from [configs/model.yaml](configs/model.yaml): amount limit, card bursts in 1h and 24h, new card with a high amount, card blacklist, device blacklist. Score = sum of weights. |
| Logistic regression | Engineered features plus C, D and M columns. Signed-log transform, median imputation with missing-value flags, rare categories grouped. |

Metrics: PR-AUC, ROC-AUC, recall and precision at 1% FPR, Brier score and log loss. The test month is not used until the final evaluation.

## LightGBM and the decision layer (week 4)

`fraudguard train lightgbm` trains the main model and decides what to do with its scores.

**Tuning.** Optuna (TPE sampler, median pruning, seed 42) searches learning rate, leaves, minimum leaf size, feature and row sampling, L1/L2 and categorical smoothing. It never sees the validation month. The training months are cut into four time chunks. Each fold trains on the earlier chunks and scores the next one, so every fold predicts the future from the past. Trials are ranked by mean PR-AUC. The final model is refitted on all training months. Defaults are 60 trials or 60 minutes, whichever comes first (`--trials` overrides the count).

**Calibration.** The policy multiplies probabilities by dollar amounts, so a score of 0.3 has to mean about 30% fraud. Isotonic regression is fitted on the validation month. Its quality is measured out-of-fold: each of five time blocks is calibrated by the other four. The report shows expected calibration error and a reliability table before and after. The bins are log-spaced, because most transactions have small probabilities.

**Cost model** ([configs/model.yaml](configs/model.yaml)). These costs are **assumptions, not merchant data**:

| Outcome | Cost |
|---|---|
| Fraud approved | the transaction amount (the merchant pays the chargeback) |
| Genuine customer declined | $15 (lost margin plus churn risk) |
| Sent to an analyst | $3 per case, at most 200 cases a day |

**Policy** ([`decision/policy.py`](src/fraudguard/decision/policy.py)). For calibrated probability p and amount A, the expected costs are:
- approve: p·A
- decline: (1−p)·$15
- review: $3

The cheapest action wins. This makes the rule amount-aware: a $2,000 transaction is declined above about 0.7% fraud probability, a $25 one only above 37.5%. The review band is wherever an analyst costs less than either mistake. When a day has more review candidates than capacity, the cases where review saves the most go first, and the rest get the cheaper of approve and decline.

On validation, a sweep of an odds multiplier checks whether to trust the calibration. A best multiplier near 1 means the probabilities are honest. The rules baseline gets the same chance: its review and decline thresholds are chosen to minimise cost on the same data.

**Sensitivity.** The false-decline and review costs are guesses, so the whole comparison is repeated with each one halved and doubled (a 3 × 3 grid). Results are in `reports/decision.md` and `reports/champion.json`, alongside cost curves for the console's cost explorer.

**Registry.** The model is stored as plain files: the LightGBM text model and a JSON file with the feature list, category vocabulary, calibrator and policy. It is logged to MLflow as code, so nothing is pickled, and registered as `fraudguard`. The `champion` alias moves to the new version only if it beats the rules baseline on cost. The scoring API will load `models:/fraudguard@champion`.

On 20,000 synthetic rows (validation month; the real dataset will be harder):

| | PR-AUC | Recall at 1% FPR | Cost per 10k transactions |
|---|---:|---:|---:|
| Rules (best thresholds) | 0.573 | 44.3% | $13,400 |
| Logistic regression | 0.857 | 84.4% | - |
| LightGBM + policy | 0.881 | 86.9% | $4,537 (66% less than rules) |
| Approve everything | - | - | $117,365 |

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check (strict)
uv run pre-commit install
```
