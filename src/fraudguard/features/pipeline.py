"""The `features` stage: base snapshot to feature snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from fraudguard.config import DataConfig, FeatureConfig
from fraudguard.data import pipeline as data_pipeline
from fraudguard.data import snapshot
from fraudguard.features.registry import GROUPS, META_COLUMNS, build_features, feature_columns
from fraudguard.log import get_logger

STAGE = "features"
log = get_logger(__name__)


@dataclass
class FeatureBuildResult:
    skipped: bool
    manifest: snapshot.Manifest
    n_features: int
    group_sizes: dict[str, int]


def build(data_cfg: DataConfig, feat_cfg: FeatureConfig, force: bool = False) -> FeatureBuildResult:
    root = data_cfg.paths.snapshot_dir
    base_manifest = snapshot.read_manifest(root, data_pipeline.STAGE)
    inputs = {data_pipeline.STAGE: base_manifest.digest()}
    fingerprint = feat_cfg.fingerprint()

    if not force and snapshot.is_up_to_date(root, STAGE, inputs, fingerprint):
        log.info("Snapshot '%s' is up to date; skipping", STAGE)
        manifest = snapshot.read_manifest(root, STAGE)
        meta = manifest.files["features"]
        return FeatureBuildResult(True, manifest, meta.columns - len(META_COLUMNS), {})

    base = snapshot.load_snapshot(root, data_pipeline.STAGE, "transactions")
    log.info("Building %d feature groups on %s rows", len(GROUPS), f"{base.height:,}")
    sizes = {name: GROUPS[name].build(base.head(5), feat_cfg).width for name in GROUPS}
    features = build_features(base, feat_cfg)

    manifest = snapshot.write_snapshot(root, STAGE, {"features": features}, inputs, fingerprint)
    log.info("Wrote snapshot '%s' (digest %s)", STAGE, manifest.digest())
    return FeatureBuildResult(False, manifest, len(feature_columns(features)), sizes)
