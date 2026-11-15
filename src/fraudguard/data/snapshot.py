"""Versioned Parquet snapshots with a SHA-256 manifest.

Each pipeline stage writes its outputs to data/snapshots/<stage>/ along with
manifest.json. The manifest records file hashes, row counts, schema, the
hashes of the inputs, the config fingerprint and the code version. Training
logs the manifest hash, so every model can be traced back to its exact data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fraudguard import __version__

MANIFEST = "manifest.json"
_CHUNK = 1 << 20


class SnapshotError(RuntimeError):
    """Raised when a snapshot is missing or does not match its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


@dataclass
class FileEntry:
    sha256: str
    rows: int
    columns: int
    schema: dict[str, str]


@dataclass
class Manifest:
    stage: str
    created_at: str
    code_version: str
    config_fingerprint: str
    inputs: dict[str, str]
    files: dict[str, FileEntry] = field(default_factory=dict)

    def digest(self) -> str:
        """One hash identifying this snapshot: logged with every model trained on it."""
        payload = json.dumps({k: v.sha256 for k, v in sorted(self.files.items())}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_path(cls, path: Path) -> Manifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["files"] = {k: FileEntry(**v) for k, v in raw["files"].items()}
        return cls(**raw)


def stage_dir(snapshot_root: Path, stage: str) -> Path:
    return snapshot_root / stage


def write_snapshot(
    snapshot_root: Path,
    stage: str,
    frames: dict[str, pl.DataFrame],
    inputs: dict[str, str],
    config_fingerprint: str,
) -> Manifest:
    """Write frames atomically: build in a temp dir, then swap it in."""
    final = stage_dir(snapshot_root, stage)
    tmp = final.with_name(f".{stage}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    manifest = Manifest(
        stage=stage,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        code_version=f"{__version__}+{git_commit()}",
        config_fingerprint=config_fingerprint,
        inputs=dict(sorted(inputs.items())),
    )
    for name, df in frames.items():
        path = tmp / f"{name}.parquet"
        df.write_parquet(path, compression="zstd", statistics=True)
        manifest.files[name] = FileEntry(
            sha256=sha256_file(path),
            rows=df.height,
            columns=df.width,
            schema={c: str(t) for c, t in df.schema.items()},
        )
    (tmp / MANIFEST).write_text(manifest.to_json(), encoding="utf-8")

    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return manifest


def read_manifest(snapshot_root: Path, stage: str) -> Manifest:
    path = stage_dir(snapshot_root, stage) / MANIFEST
    if not path.exists():
        raise SnapshotError(f"no snapshot for stage '{stage}' at {path.parent}")
    return Manifest.from_path(path)


def load_snapshot(snapshot_root: Path, stage: str, name: str, verify: bool = True) -> pl.DataFrame:
    manifest = read_manifest(snapshot_root, stage)
    if name not in manifest.files:
        raise SnapshotError(f"snapshot '{stage}' has no file '{name}'")
    path = stage_dir(snapshot_root, stage) / f"{name}.parquet"
    if verify and sha256_file(path) != manifest.files[name].sha256:
        raise SnapshotError(f"{path} does not match its manifest hash; rebuild with `fraudguard data build`")
    return pl.read_parquet(path)


def is_up_to_date(snapshot_root: Path, stage: str, inputs: dict[str, str], config_fingerprint: str) -> bool:
    try:
        manifest = read_manifest(snapshot_root, stage)
    except SnapshotError:
        return False
    if manifest.inputs != dict(sorted(inputs.items())) or manifest.config_fingerprint != config_fingerprint:
        return False
    folder = stage_dir(snapshot_root, stage)
    return all((folder / f"{n}.parquet").exists() for n in manifest.files)
