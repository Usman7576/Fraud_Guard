# Demo script (about 4 minutes)

A walkthrough for recording the project's demo video. Everything runs locally.

## Before recording

```bash
uv run fraudguard all --synthetic --web            # once, on a fresh clone (or after getting the Kaggle data)
uv run fraudguard db add-user <you> --role admin   # your own login; choose the password at the prompt
uv run fraudguard db add-api-key landing-demo --scope demo
#   put the printed key in web/.env.local as FRAUDGUARD_DEMO_API_KEY=...
```

For populated Monitoring and Overview pages, replay the later months on a separate database first (README, "Monitoring and retraining"). Then run, in two terminals:

```bash
uv run fraudguard serve
npm --prefix web run start      # http://localhost:3000
```

Use a 1440×900 browser window, and close other tabs so the 3D scenes run smoothly.

## Script

| Time | Screen | Say |
|---|---|---|
| 0:00 | Landing page hero | "FraudGuard scores card payments in real time and decides approve, review or decline by what each mistake costs. The red payments are fraud, stopped by the shield." |
| 0:20 | Scroll through **Entity graph** | "Fraud travels in rings. A chargeback on card A reaches the device and email it used, then three more cards. That two-hop count is the model's most important feature." |
| 0:45 | **Live demo**: pick *Risky first purchase*, score it | "This calls the real model. The needle shows the calibrated chance of fraud on a log scale; the decision comes from the cost policy, and these are the top three reasons." |
| 1:10 | **Results** | "Measured on a month the model never saw, with 95% intervals from a bootstrap over days, against hand-written rules." |
| 1:30 | Sign in → **Overview** | "The analyst console: weekly volume, decline and review rates, recall once chargebacks arrive, and the latest drift check." |
| 1:50 | **Review queue** → open a case | "Cases sorted by risk. Each one shows the reasons, SHAP contributions and the card's neighbourhood. A verdict closes the case and becomes a label." |
| 2:30 | **Monitoring** | "A replay worker plays later months through the live API. Drift is checked every week, and a challenger is promoted only if it is clearly better at no extra cost." |
| 3:00 | **Cost explorer**: drag *False decline cost* | "What if a false decline costs $40, not $15? The bill and the thresholds update instantly on the validation month. Nothing changes until an admin applies it." |
| 3:30 | **Settings** | "Users and roles, API keys with scopes shown once, and an audit log of every change." |
| 3:50 | Back to the landing page | "Fully local: FastAPI, MongoDB, LightGBM and Next.js, no Docker." |

## Tips

- Record at 1080p or higher. On Windows, the Xbox Game Bar (Win+Alt+R) records the active window.
- If the scenes stutter while recording, lower the window size; the canvases cap the pixel ratio at 2.
- With the synthetic data the figures are labelled as such on the page. Rebuild on IEEE-CIS (`uv run fraudguard all --force`) for the real ones.
