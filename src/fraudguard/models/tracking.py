"""MLflow tracking helpers. Every training run records what data and code it saw."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow

from fraudguard import __version__
from fraudguard.data.snapshot import git_commit


@contextmanager
def run(
    tracking_uri: str,
    experiment: str,
    run_name: str,
    tags: dict[str, str],
    params: dict[str, Any],
) -> Iterator[mlflow.ActiveRun]:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({"code_version": f"{__version__}+{git_commit()}", **tags})
        mlflow.log_params({k: _param(v) for k, v in params.items()})
        yield active


def _param(value: Any) -> str | float | int:
    if isinstance(value, int | float | str):
        return value
    text = str(value)
    return text if len(text) <= 500 else text[:497] + "..."


def log_metrics(prefix: str, metrics: dict[str, float]) -> None:
    mlflow.log_metrics({f"{prefix}_{k}": float(v) for k, v in metrics.items()})
