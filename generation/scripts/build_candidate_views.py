from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from candidate_processing import (
    QualityConfig,
    candidate_statistics,
    prepare_candidates,
    sanitize_record,
    select_diverse,
    select_oracle_balanced,
    select_random,
    write_jsonl,
)


LABEL_BLIND_VIEWS = {"all_usable", "random_4", "diverse_4"}
KNOWN_VIEWS = LABEL_BLIND_VIEWS | {"oracle_balanced_4"}
DEFAULT_VIEWS = ("all_usable", "random_4", "diverse_4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build named candidate datasets and separate gold-based analysis artifacts."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Pool containing generated/*.jsonl."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Experiment root.")
    parser.add_argument(
        "--views", nargs="+", default=list(DEFAULT_VIEWS), choices=sorted(KNOWN_VIEWS)
    )
    parser.add_argument(
        "--size", type=int, default=4, help="Candidate count for four-candidate views."
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--min-correct", type=int, default=1)
    parser.add_argument("--max-correct", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--require-parseable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-gold-in-views",
        action="store_true",
        help=(
            "Audit-only option. By default all view files omit gold answers and candidate "
            "correctness."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.size < 1:
        parser.error("--size must be >= 1")
    if args.min_correct < 0 or args.max_correct < args.min_correct:
        parser.error("--min-correct and --max-correct are inconsistent")
    return args


def main() -> None:
    args = parse_args()
    generated_dir = args.input_dir / "generated"
    if not generated_dir.exists():
        raise FileNotFoundError(f"Missing pooled generated directory: {generated_dir}")

    quality = QualityConfig(args.min_chars, args.max_chars, args.require_parseable)
    config = {
        "input_dir": str(args.input_dir),
        "views": args.views,
        "size": args.size,
        "seed": args.seed,
        "min_correct": args.min_correct,
        "max_correct": args.max_correct,
        "include_gold_in_views": args.include_gold_in_views,
        "quality": asdict(quality),
    }
    views_root = args.output_dir / "candidate_views"
    analysis_root = args.output_dir / "analysis"
    views_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    (views_root / "view_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    language_summaries = []
    for input_path in sorted(generated_dir.glob("*.jsonl")):
        language_summaries.append(process_language(args, quality, input_path))
        write_language_summary(analysis_root, language_summaries)
    write_language_summary(analysis_root, language_summaries)
    print(f"Done. Candidate views: {views_root}")
    print(f"Gold-based statistics: {analysis_root}")


def process_language(
    args: argparse.Namespace, quality: QualityConfig, input_path: Path
) -> dict[str, Any]:
    language = input_path.stem
    output_paths = {
        view: args.output_dir / "candidate_views" / view / f"{language}.jsonl"
        for view in args.views
    }
    stats_path = args.output_dir / "analysis" / "item_candidate_stats" / f"{language}.jsonl"
    all_paths = [*output_paths.values(), stats_path]
    if not args.overwrite and all(path.exists() for path in all_paths):
        return summarize_stats_file(language, stats_path, output_paths)

    view_rows: dict[str, list[dict[str, Any]]] = {view: [] for view in args.views}
    stats_rows = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            annotated, usable = prepare_candidates(record.get("candidates", []), quality)
            stats = candidate_statistics(record, annotated, usable)
            stats_rows.append(stats)
            for view in args.views:
                selected = candidates_for_view(view, usable, args, str(record.get("id")))
                output = sanitize_record(record, selected, include_gold=args.include_gold_in_views)
                output["view_metadata"] = {
                    "view_name": view,
                    "label_blind_selection": view in LABEL_BLIND_VIEWS,
                    "usable_available": len(usable),
                    "candidate_count": len(selected),
                    "requested_count": None if view == "all_usable" else args.size,
                    "complete": view == "all_usable" or len(selected) == args.size,
                    "selection_seed": args.seed if view in {"random_4", "diverse_4"} else None,
                }
                view_rows[view].append(output)

    for view, rows in view_rows.items():
        write_jsonl(output_paths[view], rows)
    write_jsonl(stats_path, stats_rows)
    return summarize_stats(language, stats_rows, output_paths)


def candidates_for_view(
    view: str, usable: list[dict[str, Any]], args: argparse.Namespace, record_id: str
) -> list[dict[str, Any]]:
    if view == "all_usable":
        return usable
    if view == "random_4":
        return select_random(usable, args.size, args.seed, record_id)
    if view == "diverse_4":
        return select_diverse(usable, args.size, args.seed, record_id)
    if view == "oracle_balanced_4":
        return select_oracle_balanced(usable, args.size, args.min_correct, args.max_correct)
    raise ValueError(f"Unknown candidate view: {view}")


def summarize_stats(
    language: str, rows: list[dict[str, Any]], output_paths: dict[str, Path]
) -> dict[str, Any]:
    raw = sum(int(row["raw_candidate_count"]) for row in rows)
    quality_pass = sum(int(row["quality_pass_candidate_count"]) for row in rows)
    usable = sum(int(row["usable_candidate_count"]) for row in rows)
    correct = sum(int(row["correct_usable_count"]) for row in rows)
    incorrect = sum(int(row["incorrect_usable_count"]) for row in rows)
    summary = {
        "language": language,
        "items": len(rows),
        "raw_candidates": raw,
        "quality_pass_candidates": quality_pass,
        "usable_candidates": usable,
        "duplicates_removed": sum(int(row["duplicates_removed"]) for row in rows),
        "correct_usable_candidates": correct,
        "incorrect_usable_candidates": incorrect,
        "usable_accuracy": correct / usable if usable else None,
        "items_with_no_usable": sum(int(row["usable_candidate_count"] == 0) for row in rows),
        "items_with_fewer_than_4_usable": sum(
            int(row["usable_candidate_count"] < 4) for row in rows
        ),
        "items_with_no_correct": sum(int(row["correct_usable_count"] == 0) for row in rows),
        "items_with_no_incorrect": sum(int(row["incorrect_usable_count"] == 0) for row in rows),
        "mean_distinct_predicted_answers": (
            sum(int(row["distinct_predicted_answers"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "view_files": {view: str(path) for view, path in output_paths.items()},
    }
    summary["regime_stats"] = aggregate_grouped_stats(rows, "regime_stats")
    summary["source_stats"] = aggregate_grouped_stats(rows, "source_stats")
    return summary


def aggregate_grouped_stats(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int | float | None]]:
    aggregated: dict[str, dict[str, int | float | None]] = {}
    for row in rows:
        for group, values in row.get(field, {}).items():
            target = aggregated.setdefault(
                group,
                {
                    "raw_candidates": 0,
                    "usable_candidates": 0,
                    "correct_usable_candidates": 0,
                    "incorrect_usable_candidates": 0,
                },
            )
            for key, value in values.items():
                target[key] = int(target.get(key) or 0) + int(value)
    for target in aggregated.values():
        usable = int(target["usable_candidates"] or 0)
        correct = int(target["correct_usable_candidates"] or 0)
        target["usable_accuracy"] = correct / usable if usable else None
    return aggregated


def summarize_stats_file(
    language: str, stats_path: Path, output_paths: dict[str, Path]
) -> dict[str, Any]:
    with stats_path.open(encoding="utf-8") as handle:
        return summarize_stats(language, [json.loads(line) for line in handle], output_paths)


def write_language_summary(analysis_root: Path, summaries: list[dict[str, Any]]) -> None:
    rows = sorted(summaries, key=lambda row: row["language"])
    json_path = analysis_root / "candidate_stats_by_language.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    nested_fields = {"view_files", "regime_stats", "source_stats"}
    csv_rows = [
        {key: value for key, value in row.items() if key not in nested_fields} for row in rows
    ]
    if not csv_rows:
        return
    with (analysis_root / "candidate_stats_by_language.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_grouped_summary(
        analysis_root / "candidate_stats_by_regime.csv", rows, "regime_stats", "regime"
    )
    write_grouped_summary(
        analysis_root / "candidate_stats_by_source.csv", rows, "source_stats", "source"
    )


def write_grouped_summary(
    path: Path, summaries: list[dict[str, Any]], field: str, group_column: str
) -> None:
    output_rows = []
    for summary in summaries:
        for group, values in summary.get(field, {}).items():
            output_rows.append({"language": summary["language"], group_column: group, **values})
    if not output_rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
