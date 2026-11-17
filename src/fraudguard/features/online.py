"""Online features: the same definitions as the offline pipeline, one transaction at a time.

The scoring API cannot rerun Polars over the whole history for every request,
so this store keeps each entity's history in memory and answers the same
questions incrementally:

- velocity: count and amount per entity over [t - window, t), from sorted
  timestamps and prefix sums (binary search, O(log n));
- behaviour: the card's earlier amounts and times of day;
- graph: the shared `EntityGraph`, exactly as in training.

"Strictly earlier" is kept the same way as offline: a transaction never sees
another one from the same second. Velocity and behaviour exclude equal
timestamps by binary search; graph updates for a second are buffered and only
applied once a later second arrives.

Transactions must arrive in time order (equal timestamps are fine). The
parity test (tests/test_online.py) replays the test month through this store
and checks every feature against the offline snapshot.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from fraudguard.config import FeatureConfig
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.data.schema import AMOUNT, DT, ID, LABEL
from fraudguard.features.graph import ENTITY_SOURCES, EntityGraph

Row = dict[str, Any]
SECONDS_PER_HOUR = 3_600


class OutOfOrderError(ValueError):
    """A transaction is older than one the store has already seen."""


_DURATION_UNITS = {"s": 1, "m": 60, "h": 3_600, "d": SECONDS_PER_DAY, "w": 7 * SECONDS_PER_DAY}


def window_seconds(window: str) -> int:
    """Polars-style duration ("1h", "7d", "3650d") in seconds."""
    unit = window[-1]
    if unit not in _DURATION_UNITS or not window[:-1].isdigit():
        raise ValueError(f"unsupported window {window!r}; use a number followed by s, m, h, d or w")
    return int(window[:-1]) * _DURATION_UNITS[unit]


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _int_str(v: object) -> str | None:
    return None if _missing(v) else str(int(float(v)))  # type: ignore[arg-type]


def card_uid(row: Row, day: int) -> str:
    """Same formula as `data.identity.add_card_uid`: card1, addr1 and the account start day."""
    d1 = row.get("D1")
    start = None if _missing(d1) else str(math.floor(day - float(d1)))  # type: ignore[arg-type]
    return "_".join([_int_str(row.get("card1")) or "", _int_str(row.get("addr1")) or "na", start or "na"])


def entities(row: Row) -> dict[str, str | None]:
    """Device, email and address keys, as `graph._entity_lists` builds them."""
    out: dict[str, str | None] = {}
    for etype, column in ENTITY_SOURCES.items():
        if column is None:
            a1 = _int_str(row.get("addr1"))
            out[etype] = None if a1 is None else f"{a1}-{_int_str(row.get('addr2')) or 'na'}"
        else:
            v = row.get(column)
            out[etype] = None if _missing(v) else str(v)
    return out


@dataclass
class _History:
    """Time-ordered events for one entity key: timestamps plus prefix sums of amount."""

    dts: list[int] = field(default_factory=list)
    cum_amount: list[float] = field(default_factory=lambda: [0.0])

    def add(self, dt: int, amount: float) -> None:
        self.dts.append(dt)
        self.cum_amount.append(self.cum_amount[-1] + amount)

    def window(self, start: int, end: int) -> tuple[int, int]:
        """Index range of events with start <= dt < end."""
        return bisect_left(self.dts, start), bisect_left(self.dts, end)


@dataclass
class _CardHistory(_History):
    amounts: list[float] = field(default_factory=list)
    angles: list[float] = field(default_factory=list)

    def add_card(self, dt: int, amount: float, angle: float) -> None:
        self.add(dt, amount)
        self.amounts.append(amount)
        self.angles.append(angle)


class OnlineFeatureStore:
    def __init__(self, cfg: FeatureConfig, reference_date: date) -> None:
        self.cfg = cfg
        self.ref = datetime.combine(reference_date, datetime.min.time())
        self.windows = [(w, window_seconds(w)) for w in cfg.velocity_windows]
        self.history = window_seconds(cfg.history_window)
        gc = cfg.graph
        self.graph = EntityGraph(
            gc.label_delay_days, list(gc.hop_entities), gc.max_neighbours, gc.entity_window_days
        )
        self.velocity: dict[str, dict[str, _History]] = {
            alias: defaultdict(_History) for alias in cfg.velocity_keys
        }
        self.cards: dict[str, _CardHistory] = defaultdict(_CardHistory)
        self.last_dt: int | None = None
        # Graph updates for the current second, applied when a later second arrives.
        self._buffer: list[tuple[str, dict[str, str | None], int, bool]] = []
        self.n_seen = 0
        self.ids: set[int] = set()  # TransactionIDs already recorded

    # Time helpers ------------------------------------------------------------
    def _clock(self, dt: int) -> tuple[int, float, int]:
        """(day index, hour as float with minutes, weekday 1-7) for a TransactionDT."""
        # The reference timestamp is midnight, so time of day comes straight from the offset.
        day, secs = divmod(dt, SECONDS_PER_DAY)
        hour = secs // SECONDS_PER_HOUR + (secs % SECONDS_PER_HOUR) // 60 / 60
        return day, hour, (self.ref.weekday() + day) % 7 + 1

    def _advance(self, dt: int) -> None:
        if self.last_dt is not None and dt < self.last_dt:
            raise OutOfOrderError(f"TransactionDT {dt} is older than the latest seen ({self.last_dt})")
        if self._buffer and dt > self._buffer[0][2]:
            for card, ents, t, fraud in self._buffer:
                self.graph.add(card, ents, t, fraud)
            self._buffer.clear()
        self.graph.release_labels(dt)

    def _keys(self, row: Row, uid: str) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for alias, column in self.cfg.velocity_keys.items():
            v = uid if column == "card_uid" else row.get(column)
            out[alias] = None if _missing(v) else str(v)
        return out

    # Public API ---------------------------------------------------------------
    def features(self, row: Row, *, peek: bool = False) -> Row:
        """Every engineered feature for `row`, computed from strictly earlier transactions.

        With `peek=True` nothing in the store changes (a "what if" for the playground): the
        time-order check, the graph's pending same-second updates and label releases are skipped."""
        dt, amount = int(row[DT]), float(row[AMOUNT])
        if not peek:
            self._advance(dt)
        day, hour, weekday = self._clock(dt)
        uid = card_uid(row, day)
        out: Row = {
            "card_uid": uid,
            "amt_log": float(np.float32(math.log1p(amount))),
            "amt_cents": float(np.float32(round(amount - math.floor(amount), 3))),
            "hour": float(np.float32(hour)),
            "weekday": weekday,
        }
        for alias, key in self._keys(row, uid).items():
            hist = self.velocity[alias].get(key) if key is not None else None
            for w, secs in self.windows:
                if key is None:
                    cnt: int | None = None
                    amt: float | None = None
                elif hist is None:
                    cnt, amt = 0, 0.0
                else:
                    lo, hi = hist.window(dt - secs, dt)
                    cnt, amt = hi - lo, float(np.float32(hist.cum_amount[hi] - hist.cum_amount[lo]))
                out[f"vel_{alias}_{w}_cnt"] = cnt
                out[f"vel_{alias}_{w}_amt"] = amt
        out.update(self._behaviour(uid, dt, amount, hour))
        out.update(self.graph.score(uid, entities(row), dt))
        return out

    def _behaviour(self, uid: str, dt: int, amount: float, hour: float) -> Row:
        hist = self.cards.get(uid)
        lo, hi = hist.window(dt - self.history, dt) if hist is not None else (0, 0)
        n = hi - lo
        if hist is None or n == 0:
            return {
                "card_prior_txn": 0,
                "card_is_new": True,
                "card_secs_since_last": None,
                "card_amt_mean": None,
                "card_amt_z": None,
                "card_amt_ratio": None,
                "card_hour_dev": None,
            }
        amounts = np.asarray(hist.amounts[lo:hi])
        mean = float(amounts.mean())
        std = float(amounts.std(ddof=1)) if n > 1 else float("nan")
        angles = np.asarray(hist.angles[lo:hi])
        usual = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
        diff = abs(hour * 2 * math.pi / 24 - usual)
        enough = n >= self.cfg.min_history_for_zscore and std > 0
        return {
            "card_prior_txn": n,
            "card_is_new": False,
            "card_secs_since_last": float(np.float32(dt - hist.dts[hi - 1])),
            "card_amt_mean": float(np.float32(mean)),
            "card_amt_z": float(np.float32((amount - mean) / std)) if enough else None,
            "card_amt_ratio": float(np.float32(amount / mean)),
            "card_hour_dev": float(np.float32(min(diff, 2 * math.pi - diff) * 24 / (2 * math.pi))),
        }

    def add(self, row: Row, is_fraud: bool | None = None) -> None:
        """Record a transaction after scoring it. `is_fraud` (if known) is used after the label delay."""
        dt, amount = int(row[DT]), float(row[AMOUNT])
        self._advance(dt)
        day, hour, _ = self._clock(dt)
        uid = card_uid(row, day)
        for alias, key in self._keys(row, uid).items():
            if key is not None:
                self.velocity[alias][key].add(dt, amount)
        self.cards[uid].add_card(dt, amount, hour * 2 * math.pi / 24)
        self._buffer.append((uid, entities(row), dt, bool(is_fraud)))
        self.last_dt = dt
        self.n_seen += 1
        if (txn_id := row.get(ID)) is not None:
            self.ids.add(int(txn_id))

    def mark_fraud(self, uid: str) -> None:
        """An analyst confirmed fraud on this card: known from now on."""
        self.graph.mark_fraud(uid, self.last_dt or 0)

    def warm(self, df: pl.DataFrame, use_labels: bool = True) -> None:
        """Load history (rows in time order), e.g. the training and validation months."""
        labels = df[LABEL].to_list() if use_labels and LABEL in df.columns else [None] * df.height
        for row, label in zip(df.iter_rows(named=True), labels, strict=True):
            self.add(row, None if label is None else label == 1)
