"""Entity graph group: who shares devices, addresses and emails with whom.

Cards, devices, billing addresses and email domains are nodes; each transaction
links its card to its device, address and email. Fraud rings reuse devices and
addresses across many stolen cards, so a card that sits two hops from cards
already known to be fraudulent is far riskier than its own history suggests.

The graph is built by streaming transactions in time order. All rows sharing a
timestamp are scored against the graph as it was *before* that timestamp and
only then added to it, matching the strictly-earlier rule used everywhere else.

Fraud labels are not known when a transaction happens: a chargeback arrives
weeks later. A fraudulent transaction only marks its card as known fraud once
`label_delay_days` have passed, so the features never use a label the business
could not have had at the time.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict, deque

import polars as pl

from fraudguard.config import FeatureConfig
from fraudguard.data.identity import UID
from fraudguard.data.load import SECONDS_PER_DAY
from fraudguard.data.schema import DT, LABEL

ENTITY_SOURCES = {"device": "DeviceInfo", "email": "P_emaildomain", "address": None}
ENTITY_TYPES = tuple(ENTITY_SOURCES)

COLUMNS = (
    "graph_card_n_devices",
    "graph_card_n_addresses",
    "graph_card_n_emails",
    "graph_device_n_cards_recent",
    "graph_address_n_cards_recent",
    "graph_card_known_fraud",
    "graph_device_fraud_cards",
    "graph_address_fraud_cards",
    "graph_2hop_cards",
    "graph_2hop_fraud_cards",
)


def address_key(df: pl.DataFrame) -> pl.Series:
    """Billing address as addr1 (region) plus addr2 (country); null when addr1 is unknown."""
    if "addr1" not in df.columns:
        return pl.Series("address", [None] * df.height, dtype=pl.String)
    addr1 = pl.col("addr1").cast(pl.Float64).fill_nan(None).cast(pl.Int64).cast(pl.String)
    if "addr2" in df.columns:
        addr2 = pl.col("addr2").cast(pl.Float64).fill_nan(None).cast(pl.Int64).cast(pl.String).fill_null("na")
    else:
        addr2 = pl.lit("na")
    return df.select(pl.when(addr1.is_not_null()).then(pl.concat_str([addr1, addr2], separator="-")))[
        :, 0
    ].alias("address")


def _entity_lists(df: pl.DataFrame) -> dict[str, list[str | None]]:
    out: dict[str, list[str | None]] = {}
    for etype, column in ENTITY_SOURCES.items():
        if column is None:
            out[etype] = address_key(df).to_list()
        elif column in df.columns:
            out[etype] = df[column].cast(pl.String).to_list()
        else:
            out[etype] = [None] * df.height
    return out


Entities = dict[str, str | None]


class EntityGraph:
    """Incremental card-entity graph. Used offline here and online by the scoring API."""

    def __init__(
        self, label_delay_days: int, hop_entities: list[str], max_neighbours: int, entity_window_days: int
    ) -> None:
        self.delay = label_delay_days * SECONDS_PER_DAY
        self.window = entity_window_days * SECONDS_PER_DAY
        self.hop_entities = hop_entities
        self.max_neighbours = max_neighbours
        # Rolling distinct-card count per device/address: event queue plus per-card counts.
        self.recent_events: dict[tuple[str, str], deque[tuple[int, str]]] = defaultdict(deque)
        self.recent_cards: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.card_links: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: {t: set() for t in ENTITY_TYPES}
        )
        self.entity_cards: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.card_fraud_known: Counter[str] = Counter()
        self.fraud_cards: set[str] = set()
        self.pending: list[tuple[int, str]] = []  # heap of (label known at, card)

    def release_labels(self, now: int) -> None:
        """Labels whose chargeback has arrived by `now` become known."""
        while self.pending and self.pending[0][0] <= now:
            _, card = heapq.heappop(self.pending)
            self.card_fraud_known[card] += 1
            self.fraud_cards.add(card)

    def mark_fraud(self, card: str, known_at: int) -> None:
        """A fraud label that becomes known at `known_at` (an analyst verdict or a chargeback)."""
        heapq.heappush(self.pending, (known_at, card))

    def neighbourhood(self, card: str) -> dict[str, list[dict[str, object]]]:
        """The card's devices and addresses, with how many cards and known-fraud cards share each."""
        links = self.card_links.get(card)
        out: dict[str, list[dict[str, object]]] = {}
        for etype in ("device", "address", "email"):
            out[etype] = [
                {
                    "id": ent,
                    "cards": len(self._members(etype, ent)),
                    "fraud_cards": len((self._members(etype, ent) & self.fraud_cards) - {card}),
                }
                for ent in sorted(links[etype] if links else ())
            ]
        return out

    def _members(self, etype: str, ent: str) -> set[str]:
        return self.entity_cards.get((etype, ent)) or set()

    def _recent_distinct(self, key: tuple[str, str], now: int) -> int:
        """Distinct cards on an entity in [now - window, now). Expired events are evicted lazily."""
        events, counts = self.recent_events.get(key), self.recent_cards.get(key)
        if events is None or counts is None:
            return 0
        while events and events[0][0] < now - self.window:
            _, old = events.popleft()
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        return len(counts)

    def score(self, card: str, entities: Entities, now: int) -> dict[str, int | None]:
        """Features for a transaction at time `now`, read from the graph as it stands."""
        links = self.card_links.get(card)
        out: dict[str, int | None] = {
            "graph_card_n_devices": len(links["device"]) if links else 0,
            "graph_card_n_addresses": len(links["address"]) if links else 0,
            "graph_card_n_emails": len(links["email"]) if links else 0,
            "graph_card_known_fraud": self.card_fraud_known[card],
        }
        for etype in ("device", "address"):
            ent = entities.get(etype)
            if ent is None:
                out[f"graph_{etype}_n_cards_recent"] = None
                out[f"graph_{etype}_fraud_cards"] = None
                continue
            out[f"graph_{etype}_n_cards_recent"] = self._recent_distinct((etype, ent), now)
            out[f"graph_{etype}_fraud_cards"] = len((self._members(etype, ent) & self.fraud_cards) - {card})

        neighbours: set[str] = set()
        for etype in self.hop_entities:
            linked = set(links[etype]) if links else set()
            if (current := entities.get(etype)) is not None:
                linked.add(current)
            for ent in linked:
                members = self._members(etype, ent)
                if len(members) <= self.max_neighbours:
                    neighbours |= members
        neighbours.discard(card)
        out["graph_2hop_cards"] = len(neighbours)
        out["graph_2hop_fraud_cards"] = len(neighbours & self.fraud_cards)
        return out

    def add(self, card: str, entities: Entities, dt: int, is_fraud: bool) -> None:
        for etype in ENTITY_TYPES:
            if (ent := entities.get(etype)) is not None:
                self.card_links[card][etype].add(ent)
                self.entity_cards[(etype, ent)].add(card)
                if etype != "email":
                    self.recent_events[(etype, ent)].append((dt, card))
                    self.recent_cards[(etype, ent)][card] += 1
        if is_fraud:
            self.mark_fraud(card, dt + self.delay)


def build(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    gc = cfg.graph
    g = EntityGraph(gc.label_delay_days, list(gc.hop_entities), gc.max_neighbours, gc.entity_window_days)
    n = df.height
    cards: list[str] = df[UID].to_list()
    dts = df[DT].to_numpy()
    labels = df[LABEL].to_numpy()
    ent_lists = _entity_lists(df)
    rows: list[Entities] = [{t: ent_lists[t][k] for t in ENTITY_TYPES} for k in range(n)]
    out: dict[str, list[int | None]] = {c: [None] * n for c in COLUMNS}

    i = 0
    while i < n:
        t = int(dts[i])
        j = i
        while j < n and dts[j] == t:
            j += 1
        g.release_labels(t)
        # Score every row at this timestamp first, then add them: same-second rows never see each other.
        for k in range(i, j):
            for name, value in g.score(cards[k], rows[k], t).items():
                out[name][k] = value
        for k in range(i, j):
            g.add(cards[k], rows[k], t, bool(labels[k] == 1))
        i = j

    return pl.DataFrame({c: pl.Series(c, out[c], dtype=pl.Int32) for c in COLUMNS})
