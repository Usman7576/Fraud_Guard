"""Leakage harness: prove that no feature looks at future rows.

Idea: if a feature only uses the past, deleting every row after time c cannot
change its value on any row at or before c. We cut the data at several points,
rebuild the features on the truncated data, and compare every surviving row
with the full-data build. Any difference is a leak.

This catches the classic mistakes: group-wide means, forward shifts, rolling
windows that include the current or later rows, and global normalisation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from fraudguard.data.schema import DT, ID

Builder = Callable[[pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True)
class Leak:
    column: str
    cutoff_dt: int
    mismatches: int
    example_id: int

    def __str__(self) -> str:
        return (
            f"{self.column}: {self.mismatches} rows change when rows after TransactionDT={self.cutoff_dt} "
            f"are removed (e.g. TransactionID {self.example_id})"
        )


def _differs(a: pl.Expr, b: pl.Expr, rtol: float) -> pl.Expr:
    both_null = a.is_null() & b.is_null()
    one_null = a.is_null() ^ b.is_null()
    if rtol > 0:
        close = (a.cast(pl.Float64) - b.cast(pl.Float64)).abs() <= rtol * (1 + b.cast(pl.Float64).abs())
        both_nan = a.cast(pl.Float64).is_nan() & b.cast(pl.Float64).is_nan()
        return ~both_null & (one_null | ~(close | both_nan).fill_null(False))
    return ~both_null & (one_null | (a != b).fill_null(True))


def find_leaks(
    df: pl.DataFrame,
    build: Builder,
    n_cutoffs: int = 5,
    rtol: float = 1e-6,
    columns: list[str] | None = None,
) -> list[Leak]:
    """Return every (column, cutoff) where truncating the future changes past values."""
    full = build(df)
    cols = columns or [c for c in full.columns if c not in (ID, DT)]
    full = (
        full.select(ID, *cols) if ID in full.columns else pl.concat([df.select(ID), full], how="horizontal")
    )

    dts = df[DT].to_numpy()
    cutoffs = np.unique(np.quantile(dts, np.linspace(0.15, 0.85, n_cutoffs), method="lower"))

    leaks: list[Leak] = []
    for cutoff in cutoffs.astype(int):
        part_df = df.filter(pl.col(DT) <= cutoff)
        part = build(part_df)
        if ID not in part.columns:
            part = pl.concat([part_df.select(ID), part], how="horizontal")
        joined = part.select(ID, *cols).join(full.select(ID, *cols), on=ID, suffix="__full")
        for c in cols:
            numeric = joined.schema[c].is_numeric()
            bad = joined.filter(_differs(pl.col(c), pl.col(f"{c}__full"), rtol if numeric else 0.0))
            if bad.height:
                leaks.append(Leak(c, int(cutoff), bad.height, int(bad[ID][0])))
    return leaks
