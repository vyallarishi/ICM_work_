from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceDir:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge generated candidate pools across regimes.")
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="Source in name=path form, where path contains generated/*.jsonl.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Optional directory whose children each contain generated/*.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    ns = parse_args()
    sources = parse_sources(ns.source_dir, ns.runs_dir)
    if not sources:
        raise ValueError("No source directories were provided.")

    generated_dir = ns.output_dir / "generated"
    summary_dir = ns.output_dir / "summary"
    generated_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    languages = discover_languages(sources)
    summaries = []
    for language in languages:
        output_path = generated_dir / f"{language}.jsonl"
        if output_path.exists() and not ns.overwrite:
            summaries.append(summarize_pooled_file(output_path))
            continue
        summaries.append(merge_language(language, sources, output_path))
        write_summary(summary_dir, summaries)
    write_summary(summary_dir, summaries)
    write_manifest(ns.output_dir, sources)
    print(f"Done. Pooled files: {generated_dir}")


def parse_sources(source_args: list[str], runs_dir: Path | None) -> list[SourceDir]:
    sources = []
    for raw in source_args:
        if "=" not in raw:
            raise ValueError(f"--source-dir must be name=path, got: {raw}")
        name, path = raw.split("=", 1)
        sources.append(SourceDir(name=name, path=Path(path)))

    if runs_dir and runs_dir.exists():
        for child in sorted(runs_dir.iterdir()):
            if child.is_dir() and (child / "generated").exists():
                sources.append(SourceDir(name=child.name, path=child))
    return sources


def discover_languages(sources: list[SourceDir]) -> list[str]:
    languages = set()
    for source in sources:
        for path in (source.path / "generated").glob("*.jsonl"):
            languages.add(path.stem)
    return sorted(languages)


def merge_language(language: str, sources: list[SourceDir], output_path: Path) -> dict[str, Any]:
    records_by_id: dict[str, dict[str, Any]] = {}
    seen_candidates_by_id: dict[str, set[tuple[str | None, str]]] = {}
    source_counts: dict[str, int] = {}
    duplicate_candidates = 0

    for source in sources:
        input_path = source.path / "generated" / f"{language}.jsonl"
        if not input_path.exists():
            continue
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record_id = str(record.get("id"))
                pooled = records_by_id.setdefault(record_id, base_record(record))
                seen = seen_candidates_by_id.setdefault(record_id, set())
                pooled.setdefault("source_records", []).append(
                    {
                        "source_name": source.name,
                        "path": str(input_path),
                        "skipped": bool(record.get("skipped")),
                        "prompt_too_long": bool(record.get("prompt_too_long")),
                        "candidate_count": len(record.get("candidates", [])),
                    }
                )
                for candidate in record.get("candidates", []):
                    normalized = normalize_text(candidate.get("text") or "")
                    key = (candidate.get("pred_number"), normalized)
                    if key in seen:
                        duplicate_candidates += 1
                        continue
                    seen.add(key)
                    merged_candidate = {
                        **candidate,
                        "source_run": source.name,
                        "global_sample_id": (
                            f"{source.name}:{candidate.get('regime_name', record.get('regime_name'))}:"
                            f"{candidate.get('sample_index')}"
                        ),
                    }
                    pooled.setdefault("candidates", []).append(merged_candidate)
                    candidate_source = str(merged_candidate.get("candidate_source") or "unknown")
                    source_counts[candidate_source] = source_counts.get(candidate_source, 0) + 1

    with output_path.open("w", encoding="utf-8") as sink:
        for record in sorted(records_by_id.values(), key=lambda row: row.get("id", "")):
            record["skipped"] = not record.get("candidates")
            record["pooled_candidate_count"] = len(record.get("candidates", []))
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "language": language,
        "total": len(records_by_id),
        "candidates": sum(len(record.get("candidates", [])) for record in records_by_id.values()),
        "duplicates_removed": duplicate_candidates,
        "main_candidates": source_counts.get("main", 0),
        "retry_candidates": source_counts.get("retry", 0),
        "wrong_chain_ablation_candidates": source_counts.get("wrong_chain_ablation", 0),
        "pooled_file": str(output_path),
    }


def base_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        not in {
            "prompt",
            "candidates",
            "source_records",
            "pooled_candidate_count",
            "skipped",
            "regime_name",
            "candidate_source",
        }
    } | {"candidates": [], "source_records": []}


def summarize_pooled_file(path: Path) -> dict[str, Any]:
    language = path.stem
    total = 0
    candidates = 0
    source_counts = {"main": 0, "retry": 0, "wrong_chain_ablation": 0}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            for candidate in record.get("candidates", []):
                candidates += 1
                source = str(candidate.get("candidate_source") or "unknown")
                if source in source_counts:
                    source_counts[source] += 1
    return {
        "language": language,
        "total": total,
        "candidates": candidates,
        "duplicates_removed": 0,
        "main_candidates": source_counts["main"],
        "retry_candidates": source_counts["retry"],
        "wrong_chain_ablation_candidates": source_counts["wrong_chain_ablation"],
        "pooled_file": str(path),
    }


def write_summary(summary_dir: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "language",
        "total",
        "candidates",
        "duplicates_removed",
        "main_candidates",
        "retry_candidates",
        "wrong_chain_ablation_candidates",
        "pooled_file",
    ]
    sorted_summaries = sorted(summaries, key=lambda row: row["language"])
    with (summary_dir / "pool_by_language.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_summaries)
    (summary_dir / "pool_by_language.json").write_text(
        json.dumps(sorted_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, sources: list[SourceDir]) -> None:
    (output_dir / "pool_sources.json").write_text(
        json.dumps(
            [{"name": source.name, "path": str(source.path)} for source in sources],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", str(text).strip().lower())


if __name__ == "__main__":
    main()
