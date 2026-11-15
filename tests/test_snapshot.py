from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fraudguard.data import snapshot


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_roundtrip_and_manifest(tmp_path: Path) -> None:
    m = snapshot.write_snapshot(
        tmp_path, "s", {"t": _frame()}, inputs={"raw.csv": "abc"}, config_fingerprint="f1"
    )
    assert m.files["t"].rows == 3
    assert snapshot.read_manifest(tmp_path, "s").digest() == m.digest()
    assert snapshot.load_snapshot(tmp_path, "s", "t").equals(_frame())


def test_tampered_file_is_detected(tmp_path: Path) -> None:
    snapshot.write_snapshot(tmp_path, "s", {"t": _frame()}, inputs={}, config_fingerprint="f1")
    pl.DataFrame({"a": [9]}).write_parquet(tmp_path / "s" / "t.parquet")
    with pytest.raises(snapshot.SnapshotError, match="manifest hash"):
        snapshot.load_snapshot(tmp_path, "s", "t")


def test_up_to_date_depends_on_inputs_and_config(tmp_path: Path) -> None:
    snapshot.write_snapshot(
        tmp_path, "s", {"t": _frame()}, inputs={"raw.csv": "abc"}, config_fingerprint="f1"
    )
    assert snapshot.is_up_to_date(tmp_path, "s", {"raw.csv": "abc"}, "f1")
    assert not snapshot.is_up_to_date(tmp_path, "s", {"raw.csv": "changed"}, "f1")
    assert not snapshot.is_up_to_date(tmp_path, "s", {"raw.csv": "abc"}, "f2")
    assert not snapshot.is_up_to_date(tmp_path, "missing", {}, "f1")


def test_rewrite_replaces_old_snapshot(tmp_path: Path) -> None:
    snapshot.write_snapshot(tmp_path, "s", {"old": _frame()}, inputs={}, config_fingerprint="f1")
    snapshot.write_snapshot(tmp_path, "s", {"new": _frame()}, inputs={}, config_fingerprint="f1")
    assert not (tmp_path / "s" / "old.parquet").exists()
    assert set(snapshot.read_manifest(tmp_path, "s").files) == {"new"}
