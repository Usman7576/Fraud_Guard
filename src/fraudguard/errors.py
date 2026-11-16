"""Exceptions shared across modules (kept free of heavy imports so the CLI can load them cheaply)."""


class NoChampionError(RuntimeError):
    """No model holds the `champion` alias in the MLflow registry yet."""
