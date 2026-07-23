from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.snapshots import build_snapshot, promote_snapshot, rollback_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source-through", required=True)
    build.add_argument("--snapshot-root", type=Path, default=Path("data/processed/snapshots"))
    build.add_argument("sources", nargs="+", type=Path)
    promote = sub.add_parser("promote")
    promote.add_argument("snapshot", type=Path)
    promote.add_argument("--pointer", type=Path, default=Path("data/processed/promoted.json"))
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--snapshot-root", type=Path, default=Path("data/processed/snapshots"))
    rollback.add_argument("--pointer", type=Path, default=Path("data/processed/promoted.json"))
    args = parser.parse_args()
    if args.command == "build":
        result = build_snapshot(
            args.sources, args.snapshot_root, source_through_date=args.source_through
        )
    elif args.command == "promote":
        result = promote_snapshot(args.snapshot, args.pointer)
    else:
        result = rollback_snapshot(args.snapshot_root, args.pointer)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
