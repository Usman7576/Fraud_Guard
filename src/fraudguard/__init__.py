"""FraudGuard: cost-aware fraud detection."""

import os

__version__ = "0.1.0"

# MLflow settings, applied before any submodule imports MLflow.
# It prints an advert for its tracing tools on import; we don't use tracing.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Download progress bars clutter CLI output and CI logs.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
# MLflow reads the git commit through GitPython, which warns loudly when git is not on PATH.
# The commit is recorded by our own tracking tags either way.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
