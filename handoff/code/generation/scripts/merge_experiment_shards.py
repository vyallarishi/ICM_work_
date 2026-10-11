from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pool disjoint experiment shards and build candidate views.")
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = discover_sources(args.shards_root)
    if not sources:
        raise FileNotFoundError(f"No generated regime directories found under {args.shards_root}")

    pooled_dir = args.output_dir / "pooled"
    merge_cmd = [
        sys.executable,
        "scripts/merge_candidate_pools.py",
        "--output-dir",
        str(pooled_dir),
    ]
    for source_name, source_path in sources:
        merge_cmd.extend(["--source-dir", f"{source_name}={source_path}"])
    if args.overwrite:
        merge_cmd.append("--overwrite")
    run(merge_cmd)

    views_cmd = [
        sys.executable,
        "scripts/build_candidate_views.py",
        "--input-dir",
        str(pooled_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.overwrite:
        views_cmd.append("--overwrite")
    run(views_cmd)


def discover_sources(shards_root: Path) -> list[tuple[str, Path]]:
    sources = []
    for shard_dir in sorted(path for path in shards_root.glob("shard_*") if path.is_dir()):
        runs_dir = shard_dir / "runs"
        if not runs_dir.exists():
            continue
        for regime_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            if (regime_dir / "generated").exists():
                sources.append((f"{shard_dir.name}-{regime_dir.name}", regime_dir))
    return sources


def run(command: list[str]) -> None:
    print("[merge_experiment_shards] Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
