from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any


class SnapshotValidationError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(sources: list[Path]) -> dict[str, Any]:
    missing = [str(path) for path in sources if not path.exists()]
    empty = [str(path) for path in sources if path.exists() and path.stat().st_size == 0]
    duplicate_names = len({path.name for path in sources}) != len(sources)
    result = {
        "passed": not missing and not empty and not duplicate_names and bool(sources),
        "file_count": len(sources),
        "missing": missing,
        "empty": empty,
        "duplicate_names": duplicate_names,
    }
    if not result["passed"]:
        raise SnapshotValidationError(json.dumps(result, sort_keys=True))
    return result


def build_snapshot(
    sources: list[Path],
    snapshot_root: Path,
    *,
    source_through_date: str,
    schema_version: str = "1",
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    validation = validate_sources(sources)
    snapshot_id = snapshot_id or (
        f"{source_through_date}-{dt.datetime.now(dt.UTC):%H%M%S}-{uuid.uuid4().hex[:8]}"
    )
    target = snapshot_root / snapshot_id
    if target.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot_id}")
    target.mkdir(parents=True)
    files = []
    checksums = {}
    for source in sources:
        destination = target / source.name
        shutil.copy2(source, destination)
        digest = sha256(destination)
        checksums[source.name] = digest
        files.append({"name": source.name, "bytes": destination.stat().st_size, "sha256": digest})
    manifest = {
        "id": snapshot_id,
        "status": "candidate",
        "source_through_date": source_through_date,
        "schema_version": schema_version,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "files": files,
        "checksums": checksums,
        "validation": validation,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise SnapshotValidationError("manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for entry in manifest.get("files", []):
        path = snapshot_dir / entry["name"]
        if not path.exists():
            errors.append(f"missing:{entry['name']}")
        elif sha256(path) != entry["sha256"]:
            errors.append(f"checksum:{entry['name']}")
    return {"passed": not errors, "errors": errors, "id": manifest.get("id")}


def promote_snapshot(snapshot_dir: Path, pointer: Path) -> dict[str, Any]:
    verified = verify_snapshot(snapshot_dir)
    if not verified["passed"]:
        raise SnapshotValidationError(json.dumps(verified))
    previous = None
    if pointer.exists():
        previous = json.loads(pointer.read_text(encoding="utf-8")).get("current")
    state = {
        "current": snapshot_dir.name,
        "previous": previous,
        "promoted_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(pointer)
    return state


def rollback_snapshot(snapshot_root: Path, pointer: Path) -> dict[str, Any]:
    if not pointer.exists():
        raise SnapshotValidationError("no promoted snapshot exists")
    state = json.loads(pointer.read_text(encoding="utf-8"))
    previous = state.get("previous")
    if not previous:
        raise SnapshotValidationError("no previous snapshot is available")
    previous_dir = snapshot_root / previous
    verified = verify_snapshot(previous_dir)
    if not verified["passed"]:
        raise SnapshotValidationError(json.dumps(verified))
    rolled_back = {
        "current": previous,
        "previous": state.get("current"),
        "promoted_at": dt.datetime.now(dt.UTC).isoformat(),
        "rollback": True,
    }
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(json.dumps(rolled_back, indent=2), encoding="utf-8")
    temporary.replace(pointer)
    return rolled_back
