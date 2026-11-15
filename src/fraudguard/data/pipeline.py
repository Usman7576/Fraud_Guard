"""The `base` stage: raw CSVs to one validated, time-split Parquet snapshot."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from fraudguard.config import DataConfig
from fraudguard.data import identity, load, schema, snapshot, split
from fraudguard.log import get_logger

STAGE = "base"
log = get_logger(__name__)


@dataclass
class BuildResult:
    skipped: bool
    manifest: snapshot.Manifest
    summaries: list[split.SplitSummary]
    uid_consistency: float | None
    warnings: list[str]


def build_base_frame(transactions: pl.DataFrame, identity_df: pl.DataFrame, cfg: DataConfig) -> pl.DataFrame:
    """Pure transformation, kept separate from I/O so tests can call it directly."""
    df = load.join_identity(transactions, identity_df)
    df = load.add_time_columns(df, cfg.reference_date)
    df = identity.add_card_uid(df)
    df = split.assign_split(df, cfg.split)
    return load.downcast(df)


def build_base(cfg: DataConfig, force: bool = False) -> BuildResult:
    t_path = cfg.paths.raw_dir / cfg.raw_files.transactions
    i_path = cfg.paths.raw_dir / cfg.raw_files.identity
    for p in (t_path, i_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download the IEEE-CIS files from Kaggle into {cfg.paths.raw_dir}, "
                "or run `fraudguard data synth` to generate a small synthetic dataset."
            )

    log.info("Hashing raw inputs")
    inputs = {t_path.name: snapshot.sha256_file(t_path), i_path.name: snapshot.sha256_file(i_path)}
    fingerprint = cfg.fingerprint()

    if not force and snapshot.is_up_to_date(cfg.paths.snapshot_dir, STAGE, inputs, fingerprint):
        log.info("Snapshot '%s' is up to date; skipping (use --force to rebuild)", STAGE)
        manifest = snapshot.read_manifest(cfg.paths.snapshot_dir, STAGE)
        df = snapshot.load_snapshot(cfg.paths.snapshot_dir, STAGE, "transactions")
        return BuildResult(True, manifest, split.summarise(df), None, [])

    log.info("Reading %s and %s", t_path.name, i_path.name)
    transactions, identity_df = load.load_raw(t_path, i_path)

    report = schema.validate_transactions(transactions)
    schema.validate_identity(identity_df)
    log.info("Validated %s rows, fraud rate %.2f%%", f"{report.rows:,}", report.fraud_rate * 100)

    df = build_base_frame(transactions, identity_df, cfg)
    summaries = split.summarise(df)
    split.check_no_overlap(summaries)
    consistency = identity.uid_label_consistency(df.filter(pl.col(split.SPLIT) == split.Split.TRAIN.value))

    manifest = snapshot.write_snapshot(
        cfg.paths.snapshot_dir, STAGE, {"transactions": df}, inputs=inputs, config_fingerprint=fingerprint
    )
    log.info("Wrote snapshot '%s' (digest %s)", STAGE, manifest.digest())
    return BuildResult(False, manifest, summaries, consistency, report.warnings)
