from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-blind retry regimes for weak candidate pools.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Override manifest model with a local path.")
    parser.add_argument("--dataset", default=None, help="Override manifest dataset with a local path.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-views", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    ns = parse_args()
    manifest = load_manifest(ns.manifest)
    stats_dir = ns.base_dir / "analysis" / "item_candidate_stats"
    if not stats_dir.exists():
        raise FileNotFoundError(
            f"Missing item statistics: {stats_dir}. Run build_candidate_views.py first."
        )

    retry_specs = build_retry_specs(stats_dir, manifest)
    (ns.base_dir / "retry_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not retry_specs:
        print("No retry items found.")
        return

    retry_dirs = []
    for regime in manifest["regimes"]:
        retry_name = str(regime["name"])
        filter_dir = retry_specs.get(retry_name)
        if filter_dir is None:
            print(f"Skipping {retry_name}: no matching retry ids.")
            continue
        run_dir = ns.base_dir / "retry" / retry_name
        run_retry_generation(ns, manifest, regime, run_dir, filter_dir)
        retry_dirs.append(run_dir)

    source_dirs = discover_source_dirs(ns.base_dir, retry_dirs)
    run_merge(ns.base_dir / "pooled", source_dirs, overwrite=True)
    if ns.build_views:
        run_build_views(ns.base_dir, ns.base_dir / "pooled", overwrite=True)


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("regimes"), list) or not data["regimes"]:
        raise ValueError(f"Manifest must contain a non-empty regimes list: {path}")
    return data


def build_retry_specs(stats_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    ids_by_retry: dict[str, dict[str, set[str]]] = {
        str(regime["name"]): {} for regime in manifest["regimes"]
    }
    for path in sorted(stats_dir.glob("*.jsonl")):
        language = path.stem
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                for regime in manifest["regimes"]:
                    if should_retry(regime, record):
                        ids_by_retry[str(regime["name"])].setdefault(language, set()).add(str(record["id"]))

    filter_root = stats_dir.parent.parent / "retry_filters"
    specs = {}
    for retry_name, ids_by_language in ids_by_retry.items():
        if not ids_by_language:
            continue
        filter_dir = filter_root / retry_name
        filter_dir.mkdir(parents=True, exist_ok=True)
        for stale_filter in filter_dir.glob("*.txt"):
            stale_filter.unlink()
        for language, ids in ids_by_language.items():
            (filter_dir / f"{safe_filename(language)}.txt").write_text(
                "\n".join(sorted(ids)) + "\n",
                encoding="utf-8",
            )
        specs[retry_name] = filter_dir
    return specs


def should_retry(regime: dict[str, Any], stats: dict[str, Any]) -> bool:
    retry_when = str(regime.get("retry_when", "insufficient_usable"))
    if retry_when == "insufficient_usable":
        minimum = int(regime.get("min_usable", 4))
        return int(stats.get("usable_candidate_count", 0)) < minimum
    if retry_when == "low_answer_diversity":
        minimum = int(regime.get("min_distinct_answers", 2))
        return int(stats.get("distinct_predicted_answers", 0)) < minimum
    if retry_when == "any_quality_shortfall":
        return bool(stats.get("quality_failure_counts"))
    raise ValueError(
        f"Unknown retry_when value '{retry_when}'. Gold-derived retry triggers are intentionally "
        "unsupported in the label-blind retry pipeline."
    )


def run_retry_generation(
    ns: argparse.Namespace,
    manifest: dict[str, Any],
    regime: dict[str, Any],
    run_dir: Path,
    filter_dir: Path,
) -> None:
    defaults = manifest.get("defaults", {})
    cmd = [
        sys.executable,
        "scripts/generate_chains.py",
        "--model",
        str(ns.model or regime.get("model", defaults.get("model", "Qwen/Qwen3-8B-Base"))),
        "--dataset",
        str(ns.dataset or regime.get("dataset", defaults.get("dataset", "open-multilang/gsm8k-106"))),
        "--split",
        str(regime.get("split", defaults.get("split", "train"))),
        "--output-dir",
        str(run_dir),
        "--regime-name",
        str(regime["name"]),
        "--candidate-source",
        str(regime.get("candidate_source", "retry")),
        "--prompt-template",
        str(regime.get("prompt_template", defaults.get("prompt_template", "zero_shot_cot"))),
        "--few-shot-count",
        str(regime.get("few_shot_count", defaults.get("few_shot_count", 0))),
        "--few-shot-split",
        str(regime.get("few_shot_split", defaults.get("few_shot_split", "test"))),
        "--dtype",
        str(regime.get("dtype", defaults.get("dtype", "bfloat16"))),
        "--tensor-parallel-size",
        str(regime.get("tensor_parallel_size", defaults.get("tensor_parallel_size", detect_visible_gpu_count()))),
        "--gpu-memory-utilization",
        str(regime.get("gpu_memory_utilization", defaults.get("gpu_memory_utilization", 0.90))),
        "--max-model-len",
        str(regime.get("max_model_len", defaults.get("max_model_len", 4096))),
        "--max-new-tokens",
        str(regime.get("max_new_tokens", defaults.get("max_new_tokens", 768))),
        "--temperature",
        str(regime.get("temperature", defaults.get("temperature", 0.7))),
        "--top-p",
        str(regime.get("top_p", defaults.get("top_p", 0.95))),
        "--n",
        str(regime.get("n", defaults.get("n", 4))),
        "--batch-size",
        str(regime.get("batch_size", defaults.get("batch_size", 128))),
        "--seed",
        str(regime.get("seed", defaults.get("seed", 1000))),
        "--tolerance",
        str(regime.get("tolerance", defaults.get("tolerance", "0"))),
        "--id-filter-dir",
        str(filter_dir),
    ]
    append_optional_flags(cmd, regime, defaults)
    languages = sorted(path.stem for path in filter_dir.glob("*.txt"))
    if languages:
        cmd.extend(["--languages", *languages])
    if ns.overwrite:
        cmd.append("--overwrite")
    print_command(cmd)
    subprocess.run(cmd, check=True)


def append_optional_flags(cmd: list[str], regime: dict[str, Any], defaults: dict[str, Any]) -> None:
    if regime.get("english_prompt_only", defaults.get("english_prompt_only", False)):
        cmd.append("--english-prompt-only")
    if regime.get("trust_remote_code", defaults.get("trust_remote_code", False)):
        cmd.append("--trust-remote-code")
    quantization = regime.get("quantization", defaults.get("quantization"))
    if quantization:
        cmd.extend(["--quantization", str(quantization)])
    max_num_seqs = regime.get("max_num_seqs", defaults.get("max_num_seqs"))
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    max_num_batched_tokens = regime.get(
        "max_num_batched_tokens", defaults.get("max_num_batched_tokens")
    )
    if max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
    if not regime.get("enable_prefix_caching", defaults.get("enable_prefix_caching", True)):
        cmd.append("--no-enable-prefix-caching")
    preamble_file = regime.get("assistant_preamble_file", defaults.get("assistant_preamble_file"))
    if preamble_file:
        cmd.extend(["--assistant-preamble-file", str(preamble_file)])


def discover_source_dirs(base_dir: Path, retry_dirs: list[Path]) -> list[Path]:
    source_dirs = []
    runs_dir = base_dir / "runs"
    if runs_dir.exists():
        source_dirs.extend(sorted(child for child in runs_dir.iterdir() if (child / "generated").exists()))
    pool_sources_path = base_dir / "pooled" / "pool_sources.json"
    if pool_sources_path.exists():
        pool_sources = json.loads(pool_sources_path.read_text(encoding="utf-8"))
        source_dirs.extend(
            Path(source["path"])
            for source in pool_sources
            if (Path(source["path"]) / "generated").exists()
        )
    source_dirs.extend(retry_dirs)
    unique_sources = []
    seen = set()
    for source_dir in source_dirs:
        resolved = source_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_sources.append(source_dir)
    return unique_sources


def run_merge(pooled_dir: Path, source_dirs: list[Path], overwrite: bool) -> None:
    cmd = [sys.executable, "scripts/merge_candidate_pools.py", "--output-dir", str(pooled_dir)]
    for source_dir in source_dirs:
        cmd.extend(["--source-dir", f"{source_dir.name}={source_dir}"])
    if overwrite:
        cmd.append("--overwrite")
    print_command(cmd)
    subprocess.run(cmd, check=True)


def run_build_views(base_dir: Path, pooled_dir: Path, overwrite: bool) -> None:
    cmd = [
        sys.executable,
        "scripts/build_candidate_views.py",
        "--input-dir",
        str(pooled_dir),
        "--output-dir",
        str(base_dir),
    ]
    if overwrite:
        cmd.append("--overwrite")
    print_command(cmd)
    subprocess.run(cmd, check=True)


def detect_visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != "-1":
        return len([item for item in visible.split(",") if item.strip()])
    try:
        result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return 1
    count = len([line for line in result.stdout.splitlines() if line.strip()])
    return max(count, 1)


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def print_command(cmd: list[str]) -> None:
    print("[run_adaptive_retry] Running:", " ".join(shell_quote(part) for part in cmd), flush=True)


def shell_quote(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return repr(value)
    return value


if __name__ == "__main__":
    main()
