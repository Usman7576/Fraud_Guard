"""Feature group registry and the combined feature builder."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import polars as pl

from fraudguard.config import FeatureConfig
from fraudguard.data.identity import UID
from fraudguard.data.schema import DT, ID, LABEL
from fraudguard.data.split import SPLIT
from fraudguard.features import base, behaviour, graph, velocity

BuildFn = Callable[[pl.DataFrame, FeatureConfig], pl.DataFrame]

# Carried alongside features but never fed to a model.
META_COLUMNS: tuple[str, ...] = (ID, DT, "ts", SPLIT, LABEL, UID)


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    build: BuildFn
    description: str


GROUPS: dict[str, FeatureGroup] = {
    g.name: g
    for g in (
        FeatureGroup("base", base.build, "Raw columns plus per-row derivations"),
        FeatureGroup("velocity", velocity.build, "Counts and amounts per entity over 1h / 24h / 7d"),
        FeatureGroup("behaviour", behaviour.build, "How unusual this transaction is for this card"),
        FeatureGroup("graph", graph.build, "Shared devices and addresses; known fraud within two hops"),
    )
}


def build_features(df: pl.DataFrame, cfg: FeatureConfig, groups: Sequence[str] | None = None) -> pl.DataFrame:
    """Meta columns plus every requested group's columns, in the base frame's row order."""
    if not df[DT].is_sorted():
        raise ValueError("base frame must be sorted by TransactionDT")
    parts = [df.select(META_COLUMNS)]
    for name in groups or GROUPS:
        out = GROUPS[name].build(df, cfg)
        if out.height != df.height:
            raise RuntimeError(f"feature group '{name}' changed the row count")
        if clash := set(out.columns) & set().union(*(p.columns for p in parts)):
            raise RuntimeError(f"feature group '{name}' redefines columns: {sorted(clash)}")
        parts.append(out)
    return pl.concat(parts, how="horizontal")


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]
