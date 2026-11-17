# FraudGuard

Cost-aware fraud detection for online card payments: a scoring API that chooses its decision threshold by minimising business cost, explains every decision, and detects when it needs retraining. It comes with an analyst console and a 3D landing page.

Built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (590,540 e-commerce transactions, 3.5% fraud). It is evaluated on future data, in money terms, with confidence intervals.

> **Status:** all 10 weeks built: data, features, model, evaluation, API, console, monitoring, landing page and end-to-end tests. The full plan is in [docs/FraudGuard_Project_Plan.pdf](docs/FraudGuard_Project_Plan.pdf).

## Results

Measured on the **test month**, which nothing was tuned on. Intervals are 95% bootstrap intervals over whole days (1,000 resamples), and differences against the rules are paired.

| | LightGBM + cost policy | Hand-written rules |
|---|---:|---:|
| PR-AUC | **0.884** [0.828, 0.939] | 0.694 [0.591, 0.791] |
| Fraud caught at a 1% false-positive rate | **91.7%** [86.2%, 96.4%] | 84.3% [40.0%, 90.3%] |
| Cost per 10,000 payments | **$3,829** [2,091, 6,428] | $12,748 [9,676, 16,752] |
| Saving against the rules | **$8,919 per 10,000** [6,092, 12,534], 70% | |

> These figures come from the **20,000-row synthetic sample** that ships with the code (test month: 3,362 payments, 108 fraud), so the pipeline can be run without the Kaggle download. With the IEEE-CIS files in `data/raw/`, `uv run fraudguard all --force` rebuilds everything and the same tables, the landing page and `reports/` show the real figures.

Also: calibrated probabilities (expected calibration error 0.003), scoring in about 5 ms inside the API, and a replay of later months in which drift was detected weekly and two retrained challengers were tested fairly and rejected (week 8).

## Stack

| Layer | Tools |
|---|---|
| ML and data | Python 3.12, Polars, LightGBM, Optuna, SHAP, MLflow, Evidently |
| Backend | FastAPI, Pydantic v2, MongoDB |
| Frontend | Next.js, React, TypeScript, Tailwind, React Three Fiber, Framer Motion, GSAP |
| Tooling | uv, Typer CLI, ruff, mypy, pytest, Hypothesis, GitHub Actions |

Everything runs locally. There is no Docker and no GPU.

## Quick start

One command rebuilds everything on a fresh clone. It needs [uv](https://docs.astral.sh/uv/), Node.js 22 and a local MongoDB; there is no Docker.

```bash
uv run fraudguard all --synthetic --web
```

This builds the data (synthetic if the Kaggle files are missing), features, reports and the tuned, calibrated model, then evaluates it, sets up MongoDB and builds the web app. Then:

```bash
uv run fraudguard db add-user <you> --role admin   # choose your own password at the prompt
uv run fraudguard serve                            # API on :8000
npm --prefix web run start                         # web app on :3000
```

The stages can also be run one at a time:

```bash
uv sync                               # installs Python 3.12 deps from uv.lock
uv run fraudguard data synth          # optional: synthetic data, until the Kaggle files are in place
uv run fraudguard all                 # rebuild every stage
uv run fraudguard data info           # show the current snapshot manifest
uv run fraudguard features build      # build the feature snapshot
uv run fraudguard report adversarial  # train-vs-test drift report
uv run fraudguard train baselines     # rules and logistic regression
uv run fraudguard train lightgbm      # tune, calibrate, choose the policy, register the champion
uv run fraudguard report ablation     # what each feature group adds, and what it costs in drift
uv run fraudguard evaluate            # final test-month results with bootstrap intervals
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

## Final evaluation, ablation and explanations (week 5)

**Test month** (`fraudguard evaluate`). The champion is loaded from the registry, with the calibrator and policy it was registered with. The rules baseline's thresholds are fixed on validation. Nothing is chosen on the test month. Every number has a 95% bootstrap interval from 1,000 resamples of **whole days**, not single rows. Fraud comes in bursts, and resampling rows would treat one ring's afternoon as fifty independent events, making the intervals too narrow. Differences against the rules are **paired**: both are scored on the same resampled days, so shared noise cancels out.

On 20,000 synthetic rows (test month: 3,362 transactions, 108 fraud):

| Metric | LightGBM | Rules | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.884 [0.828, 0.939] | 0.694 [0.591, 0.791] | +0.191 [+0.128, +0.251] |
| Recall at 1% FPR | 91.7% [86.2%, 96.4%] | 84.3% [40.0%, 90.3%] | +7.4% [+3.0%, +50.9%] |
| Fraud $ caught | 97.6% [94.5%, 99.5%] | 94.9% [89.7%, 98.6%] | |
| Cost per 10k | $3,829 [2,091, 6,428] | $12,748 [9,676, 16,752] | saves $8,919 [6,092, 12,534] |

Cost saving vs rules: **70.0% [54.4%, 82.8%]**. Brier 0.0057, expected calibration error 0.0033. The rules' recall interval is very wide because their score takes only a few values, so the 1% FPR cut lands on large ties. `reports/evaluation.md` also has an **error analysis** of the fraud each policy lets through. It splits that fraud by:
- amount band
- product code
- device type
- email domain
- new vs returning card

**Ablation** (`fraudguard report ablation`, validation month). Feature groups are added in turn and each set is retrained with the champion's tuned parameters. *Gain* is the paired PR-AUC difference against the previous row. *Adversarial AUC* is how easily the test month can be told apart from training using only that feature set.

| Feature set | Features | PR-AUC | Gain | Adversarial AUC |
|---|---:|---:|---:|---:|
| base | 26 | 0.729 [0.660, 0.807] | - | 0.703 |
| + velocity | 44 | 0.728 [0.653, 0.803] | -0.001 [-0.027, +0.024] | 0.911 |
| + behaviour | 51 | 0.789 [0.713, 0.861] | +0.061 [+0.031, +0.088] | 0.943 |
| + graph | 61 | 0.881 [0.844, 0.922] | **+0.092 [+0.039, +0.143]** | 1.000 |
| all − lifetime counters | 55 | 0.861 [0.814, 0.914] | -0.021 [-0.043, +0.005] | 0.992 |

The entity graph is the largest single gain. Dropping the six lifetime counters costs 0.021 PR-AUC, and the interval includes zero, but it barely reduces drift (1.000 → 0.992). So the counters are not the main source of the drift, and they stay for now. On synthetic data, velocity adds nothing. These conclusions are re-checked on the real data.

**Explanations.** LightGBM computes exact TreeSHAP values natively (`pred_contrib`), so no extra library or sampling is needed, and the contributions add up exactly to the model's log-odds. [`explain/reasons.py`](src/fraudguard/explain/reasons.py) turns the top three *risk-raising* features into plain language, using the transaction's real values. For example:
- "7 earlier transactions on this card in the last 1h"
- "linked to 56 confirmed-fraud cards through shared devices or addresses"
- "amount is +3.2 standard deviations from this card's usual"

IEEE-CIS's masked columns (C, D, M, V, id_) are described as masked rather than given an invented meaning. The evaluation report lists the global mean |SHAP| and example reason codes. The scoring API will return them with every decision.

### Limits

- Most IEEE-CIS columns are masked by the data provider. Some may stand in for location or demographics, and that cannot be audited here.
- Labels are reported chargebacks; fraud that was never reported is invisible to the model.
- Costs are assumptions, not merchant data (see the sensitivity analysis).
- `card_uid` approximates a card; a real deployment would use a tokenised card ID.
- Results hold for this merchant mix and period. The drift monitoring (week 8) measures how fast they degrade.

## Scoring API on MongoDB (week 6)

```bash
uv run fraudguard db init                        # collections, JSON-schema validators, indexes
uv run fraudguard db add-user alice --role admin # prompts for a password (Argon2id hash stored)
uv run fraudguard db add-api-key replay-worker   # prints a key once; only its SHA-256 is stored
uv run fraudguard serve                          # http://localhost:8000/docs
```

MongoDB runs as the local Windows service (`mongodb://localhost:27017`, database `fraudguard`). Browse it in Compass.

| Method | Path | Who | What |
|---|---|---|---|
| POST | `/api/v1/score` | API key | Score one transaction: calibrated probability, approve / review / decline, top three reasons, model version |
| POST | `/api/v1/score/batch` | API key | Up to 1,000 transactions in time order |
| POST | `/api/v1/auth/login`, `/refresh` | public / refresh token | 15-minute access token, 7-day refresh token (JWT, HS256) |
| GET | `/api/v1/cases` | analyst | Review queue: filter by status, probability, amount and assignee; sort; paginate |
| GET / PATCH | `/api/v1/cases/{txn_id}` | analyst | Detail with features, SHAP contributions and the card's entity neighbourhood; assign, note, verdict |
| GET | `/api/v1/kpi/weekly` | analyst | Volume, decline and review rates, recall on labelled fraud, cost, p95 latency |
| GET / PUT | `/api/v1/policy` | analyst / admin | Cost assumptions and review capacity; changes apply immediately and are audited |
| GET / POST | `/api/v1/models`, `/models/{v}/promote` | admin | Registry versions with validation metrics; promote or roll back (audited) |
| GET | `/api/v1/health` | public | Model version, transactions in memory, MongoDB ping |

**Online features** ([`features/online.py`](src/fraudguard/features/online.py)). The live scorer uses the same definitions as training:
- Velocity windows use sorted timestamps with prefix sums, with lookups by binary search over `[t − window, t)`.
- Behaviour uses the card's earlier amounts and times of day.
- The graph is the same `EntityGraph` class used in training.

Rows in the same second never see each other: velocity and behaviour exclude equal timestamps, and graph updates are buffered until a later second arrives. The **parity test** replays the test month through the store and compares every engineered feature with the offline snapshot. On the 20,000-row synthetic data it finds **0 mismatches** across 3,362 transactions × 39 engineered features plus the card ID, at 0.1 ms per transaction. A second test goes end to end: it scores the test month through the HTTP API and checks that the probabilities equal the model's on the offline features.

**Latency.** On a laptop, scoring a transaction takes 2.3 ms at p95 inside the API: online features, LightGBM, exact SHAP and the decision. Over HTTP, p95 is 12.2 ms per `/score` request (280 requests, API-key lookup and MongoDB writes included). The target was 50 ms. The API test asserts the server-side p95 stays under 50 ms.

**State and storage.** For speed, the online state lives in the API's memory. At startup it is rebuilt from:
- the feature snapshot (history up to `FRAUDGUARD_SERVE_HISTORY_UNTIL`);
- every live transaction in the `transactions` collection;
- analyst fraud verdicts in `labels`.

So a restart loses nothing, and this is tested. Predictions, transactions and new cases are written in a background task after the response is sent, so a slow insert never delays a decision. A **review** decision opens a case with a 24-hour SLA. An analyst's verdict closes it and writes a label. A fraud verdict also marks the card as known fraud in the graph straight away.

| Collection | Holds | Indexes |
|---|---|---|
| `predictions` | txn, time, model version, probability, decision, reasons, latency | txn (unique), time, model version + time |
| `transactions` | raw request rows, for rebuilding online state | txn (unique), time |
| `cases` | status, assignee, verdict, notes, features, SHAP, SLA | txn (unique), status + probability, assignee + status |
| `labels` | label, source (dataset or analyst), when it became known | txn (unique), known_at |
| `kpi_weekly` | weekly KPIs, written by an aggregation with `$lookup` and `$percentile` | week |
| `users`, `api_keys`, `audit_log`, `policy` | access and admin history | username / key hash (unique), time |

**Live capacity.** Offline, a day's review candidates are ranked by how much review saves. Live, the system can't see the rest of the day, so reviews are first come, first served until the daily capacity is used. Overflow cases get the cheaper of approve and decline.

**Security.** Passwords are hashed with Argon2id, and unknown usernames take as long to reject as wrong passwords. Access and refresh tokens are not interchangeable. API keys are stored only as hashes and rate-limited per key. CORS only allows the console origin. Admin actions (policy, promotion, verdicts, user and key creation) go to `audit_log`.

## Analyst console (week 7)

```bash
cd web
npm install
npm run dev          # http://localhost:3000 (the API must be running: uv run fraudguard serve)
```

Next.js 15 (App Router), React 19, TypeScript (strict), Tailwind CSS 4 and a few hand-written UI primitives in the shadcn/ui style. There is no component library and no chart library: the charts are small SVG components.

| Page | What an analyst does there |
|---|---|
| **Overview** | Latest week's volume, decline and review rates, recall on labelled fraud and cost per 10,000, each with a sparkline. Below that, a weekly trend chart per measure, with a hover crosshair and a table view. |
| **Review queue** | Open cases sorted by risk, amount or deadline, filtered by status and minimum risk. Each row shows a risk meter, the amount, the top two reasons and the SLA. Keyboard triage: `j`/`k` to move, `Enter` to open. |
| **Case detail** | The three reasons in plain words, and SHAP contributions as diverging bars (red raises risk, blue lowers it, and the direction is also written out). Also the transaction facts, the card's entity neighbourhood (devices, addresses and emails, with confirmed-fraud links ringed in red and counted), notes, and the **Fraud / Genuine** verdict. The verdict closes the case and writes a label. |
| **Score playground** | A what-if form with presets. The calibrated probability is shown on a gauge, with the decision, the reasons and the contributions. It calls `POST /api/v1/score/preview`, which stores nothing and leaves the live feature state untouched. |

**Auth.** The browser never holds a token:
- Signing in goes through a Next.js route handler, which keeps the JWTs in `httpOnly`, `SameSite=Lax` cookies.
- Middleware guards `/console`. It refreshes the access token shortly before it expires and passes the new one to the same request.
- Interactive calls go through `/api/fg/*`. This proxy adds the bearer token, forwards only the endpoints the console uses (it does not forward admin policy changes or live `/score`), and refuses cross-site writes.
- Post-login redirects only accept same-site paths.

**Design.**
- One accent colour, neutral surfaces, and light and dark themes chosen separately.
- Decisions and statuses always pair colour with an icon and a word.
- Chart colours come from a palette validated for colour-blind readers. Lines are 2px, gridlines hairline, and text uses ink colours rather than series colours.
- The layout works down to phone width.

**Checked in a browser** against a throwaway database with the test month scored through the API (172 review cases). Sign-in redirects, KPI charts and tooltips, keyboard triage, assigning, a note and a fraud verdict, the playground, dark mode and a 375px phone layout all work. That check found and fixed three problems:
- A React effect returned the Promise that newer browsers give back from `scrollIntoView`.
- Two reason texts were misleading. Round amounts were called "currency conversion", and D1 was labelled masked although its meaning (days since the card was first seen) is known.
- The SHAP bars overflowed on phones.

## Monitoring and retraining (week 8)

```bash
# 1. An API whose history ends with the training months, on its own database:
FRAUDGUARD_SERVE_HISTORY_UNTIL=train FRAUDGUARD_MONGO_DB=fraudguard_replay uv run fraudguard serve
# 2. A key for the worker (printed once), then replay the validation and test months:
FRAUDGUARD_MONGO_DB=fraudguard_replay uv run fraudguard db add-api-key replay-worker --scope score --scope labels --scope promote
FRAUDGUARD_MONGO_DB=fraudguard_replay FRAUDGUARD_REPLAY_API_KEY=... uv run fraudguard replay
```

The replay worker ([`monitoring/replay.py`](src/fraudguard/monitoring/replay.py)) plays later months through the **live API**, week by week, as if they were production. Settings are in [configs/monitoring.yaml](configs/monitoring.yaml).

1. **Score.** Transactions go to `/score/batch` in batches of 250. Predictions and review cases are logged in MongoDB.
2. **Chargebacks.** Before each batch, labels whose time has come (transaction time + 30 days) are sent to `POST /labels`. The live entity graph learns about fraud exactly as late as a merchant would. Analyst verdicts are never overwritten.
3. **Drift.** Each week is compared with a sample of the training months on the champion's 30 most important features, using Evidently (K-S, chi-square, Wasserstein or Jensen-Shannon, chosen per column). The HTML report goes to `reports/drift/`, and a summary goes to the `drift_reports` collection.
4. **Recall.** For the latest week whose chargebacks have all arrived, recall is the share of labelled fraud that was declined or reviewed.
5. **Trigger.** Retraining starts when more than 30% of monitored features drift, or recall falls 5 points below the champion's validation recall. It runs at most once every 2 weeks.
6. **Challenger.** LightGBM is refitted with the champion's tuned parameters on a 180-day window of labelled data. Its newest 14 days are held back for isotonic calibration, and the 14 days after those for the comparison. The challenger is logged to MLflow and registered with the `challenger` alias.
7. **Promote.** Both models score the comparison days. A paired bootstrap over days measures the PR-AUC gain and the cost difference. The challenger is promoted (through `/ops/models/{v}/promote`, audited) only if the gain's lower bound is above zero **and** it costs no more.

**Fair comparisons only.** Retraining waits until the comparison days lie *after* the champion's own training data. Otherwise the champion would be scored on rows it was fitted to. The first replay without this rule rejected every challenger, but for an unfair reason. The worker now logs "skipped: N more days needed" until a fair test is possible.

**On the 20,000-row synthetic data** (validation and test months, 6,747 transactions over 10 weeks, 94–193 seconds):

| | |
|---|---|
| Chargebacks | Arrive 30 days late, so recall is first measurable in week 6 of the replay: 89–100%, against 95.1% on validation |
| Drift share | 50–77% every week. The synthetic data's lifetime counters grow with time (the same drift the week 3 adversarial validation found), so retraining was triggered at every allowed slot |
| Decisions | Weeks 1–6: skipped (not enough new labels for a fair test). Weeks 7 and 9: challengers trained and **rejected**, with PR-AUC gain −0.047 [−0.086, −0.006] and −0.045 [−0.092, −0.007] |

On this data, a model retrained on fewer, more recent rows does not beat the tuned champion, and the worker correctly keeps the champion. On real data, the drift trigger may need its own threshold. The point of the loop is that it measures before it promotes.

**API additions.**
- **Key scopes:** `score`, `labels` and `promote`. Existing keys can only score.
- **Machine endpoints:** `POST /labels` and `POST /ops/models/{v}/promote`.
- **Analyst read endpoints:** `GET /monitoring/drift`, `/monitoring/drift/{week}`, `/monitoring/drift/{week}/report` and `/monitoring/retraining`.
- **Sandboxed reports:** the Evidently HTML is served with `Content-Security-Policy: sandbox allow-scripts`. Its scripts run in an opaque origin and cannot act as the signed-in analyst.

**Console.**
- **Monitoring:** drift share against the retraining line, recall as chargebacks arrive, the decision timeline with bootstrap intervals, and the weekly drift table with links to the Evidently reports.
- **Models (admin only):** every registered version with its validation metrics and the challengers' measured gains. One click promotes or rolls back, and the API audits it.

## Landing page (week 9)

```bash
# A key for the web server's live demo (scope "demo" only: it scores and stores nothing).
uv run fraudguard db add-api-key landing-demo --scope demo
# Give it to the web server, e.g. in web/.env.local (git-ignored):
#   FRAUDGUARD_DEMO_API_KEY=...
cd web && npm run dev   # http://localhost:3000
```

The public page at `/` is always dark, because the 3D scenes glow and glow needs a dark surface. It has six sections:

| Section | What the visitor sees | How it is built |
|---|---|---|
| **Hero** | A turning globe of payments. Most arcs pulse blue end to end; a few flash red and stop where a shield ring catches them. | React Three Fiber: shader points on a Fibonacci sphere, tube arcs with a travelling-pulse shader, a fresnel rim, bloom from `@react-three/postprocessing`. Phones get the same globe as an SVG animated with CSS (see below) |
| **How it works** | Data, features, model, decision, revealed one after another | Framer Motion, staggered. The decision card quotes the live cost policy |
| **Entity graph** | Cards, devices and emails. Scrolling lights a confirmed fraud card, then what it used (one hop), then the cards sharing those (two hops), while the camera moves in | drei `Float` and `Line`. GSAP ScrollTrigger scrubs the progress, and the steps beside the scene follow it |
| **Live demo** | A payment form with presets. A 3D needle swings to the score on a log scale, then the decision and the top three reasons slide in | Real scoring through `POST /api/demo/score` (below) |
| **Results** | Recall at 1% FPR, cost saved per 10,000 payments and PR-AUC, counting up, each with its 95% interval and the rules baseline | Read from `reports/evaluation.json` at build time. The page says when the figures come from the synthetic sample |
| **Call to action** | Open the console, or read the code | Plain links |

**The live demo.** The browser posts to the web server's `/api/demo/score`:
- That route checks the request comes from the page's own origin and allows 10 tries a minute per visitor.
- It holds the demo key server-side and calls `POST /api/v1/demo/score`.
- The API endpoint needs a key with the `demo` scope and has its own limit (`FRAUDGUARD_DEMO_RATE_LIMIT_PER_MINUTE`, default 60).
- It accepts only a few plain fields: amount, product, card network and type, email domain, device, and card age.
- It scores them as a new card's first payment, right now, without storing anything. The card uses common issuer and billing values and a card number that never occurs in the data, so no real card's history is shown.

The plan said the demo would call the real `/score`. That endpoint records the payment, updates the live features and can open a review case, so the demo uses a no-write endpoint instead.

**Reason codes.** Features that read the same are now one reason (for example `TransactionAmt` and `amt_log` both say "amount of $849.00"), with their contributions added. This affects the API, the console and the evaluation report.

**Performance and accessibility.**
- SVG versions of the globe, graph and gauge paint first, so the text never waits for three.js.
- Each 3D scene loads with `next/dynamic` (`ssr: false`) only when the browser is idle and the scene is within a screen of the viewport. WebGL support is checked at that point too, because creating a test context is itself slow.
- **Phones and low-power machines keep the SVG globe**, animated with CSS on its own compositing layer. Starting three.js there blocked the main thread for seconds (measured below).
- Without WebGL the SVG versions stay. With reduced motion everything is still, the graph section becomes a plain list and the counters show their final values.
- The device pixel ratio is capped at 2, and each canvas stops rendering frames when it is off screen.

Measured in Chrome on this project's laptop (Intel Iris Xe), production build, median of three runs:

| | First paint | Largest paint | Layout shift | Blocking time | Frame rate |
|---|---:|---:|---:|---:|---:|
| Desktop, 1440×900 | 0.25 s | 0.25 s | 0 | 0.25 s | **60 fps** (globe) |
| Phone, 4× slower CPU, fast 4G | 0.9 s | 1.8 s | 0 | 2.6 s | — |

Desktop meets the plan's targets. On a throttled phone, most of the remaining blocking time is evaluating the page's React and animation code and the first layout. The official Lighthouse score was not run here; to check it, open the page in Chrome, then DevTools → Lighthouse.

## Week 10: cost explorer, settings, end-to-end tests

**Cost explorer** (admin, `/console/cost`):
- Four sliders: false-decline cost, review cost, reviews per day and the risk multiplier.
- For each setting, `POST /api/v1/policy/simulate` replays the **validation month**, scored once by the serving model and cached per version, through the decision layer. It returns the cost per 10,000 payments, the fraud stopped, and the decline and review rates, each next to the current policy.
- It also returns the cost curve as the false-decline cost moves from $5 to $100, and the probability **thresholds the policy implies** at nine amounts. A unit test checks these thresholds against the live decision function.
- Nothing changes until **Apply this policy**, which is audited.

**Settings** (admin, `/console/settings`):
- **Users:** add users, change roles, deactivate, set passwords. The API refuses to remove the last active admin, or your own admin access.
- **API keys:** create keys with scopes (`score`, `labels`, `promote`, `demo`). A key is shown once and only its hash is stored. Keys can be revoked.
- **Audit log:** paged, newest first. No passwords or hashes are ever written to it.

**Console polish.**
- **Theme:** a light/dark/system switch, remembered in the browser and applied before the first paint.
- **Overview:** a drift badge for the latest weekly check.
- **Favicon:** the app has one now.

**End-to-end tests** (Playwright, 10 tests):

```bash
cd web && npm run e2e
```

`e2e/run.mjs` does the following:
1. Creates a throwaway MongoDB database, with users whose passwords are generated for the run.
2. Starts the real API and the production web build, and scores 400 test-month payments.
3. Runs the tests in your installed Google Chrome (no browser download).
4. Stops both servers and drops the database.

The tests cover:
- The landing page on desktop (WebGL globe) and phone (SVG globe), the live demo, no horizontal scroll, and no console errors.
- The reduced-motion fallback.
- Cross-site posts to the demo, which are refused.
- Sign-in and roles, an analyst closing a case, the cost explorer, and settings.

**Security.**
- Next.js is upgraded to 15.5.29, which fixes advisories in 15.5.4, including a middleware bypass that matters here because middleware guards the console.
- `postcss` and `sharp` are pinned to patched versions through npm `overrides`. `npm audit --omit=dev` reports no vulnerabilities.

**Demo.** [docs/demo.md](docs/demo.md) is a four-minute script for recording the walkthrough video.

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check (strict)
uv run pre-commit install
cd web && npm run lint && npm run typecheck && npm test && npm run build
cd web && npm run e2e     # end-to-end (MongoDB running, model trained)
```
