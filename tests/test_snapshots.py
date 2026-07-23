from __future__ import annotations

import pytest

from pipeline.snapshots import (
    SnapshotValidationError,
    build_snapshot,
    promote_snapshot,
    rollback_snapshot,
    verify_snapshot,
)


def test_snapshot_promotion_and_rollback(tmp_path):
    source = tmp_path / "rates.json"
    source.write_text('{"version": 1}')
    root = tmp_path / "snapshots"
    first = build_snapshot(
        [source], root, source_through_date="2026-07-22", snapshot_id="first"
    )
    pointer = tmp_path / "promoted.json"
    assert promote_snapshot(root / first["id"], pointer)["current"] == "first"

    source.write_text('{"version": 2}')
    second = build_snapshot(
        [source], root, source_through_date="2026-07-23", snapshot_id="second"
    )
    state = promote_snapshot(root / second["id"], pointer)
    assert state == {
        "current": "second",
        "previous": "first",
        "promoted_at": state["promoted_at"],
    }
    assert rollback_snapshot(root, pointer)["current"] == "first"


def test_tampered_snapshot_cannot_promote(tmp_path):
    source = tmp_path / "rates.json"
    source.write_text('{"version": 1}')
    root = tmp_path / "snapshots"
    build_snapshot([source], root, source_through_date="2026-07-23", snapshot_id="candidate")
    (root / "candidate" / "rates.json").write_text("tampered")
    assert not verify_snapshot(root / "candidate")["passed"]
    with pytest.raises(SnapshotValidationError):
        promote_snapshot(root / "candidate", tmp_path / "promoted.json")
