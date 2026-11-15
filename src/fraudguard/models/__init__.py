"""Models: rules baseline, logistic regression and (next) LightGBM."""

import os

# MLflow prints an advert for its tracing tools on import; we don't use tracing.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
