from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manifest-defined generation regimes.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Override the manifest model, including a local path.")
    parser.add_argument("--dataset", default=None, help="Override the manifest dataset, including a local path.")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--limit-per-language", type=int, default=None)
    parser.add_argument("--language-shard-count", type=int, default=int(os.environ.get("LANGUAGE_SHARD_COUNT", "1")))
    parser.add_argument("--language-shard-index", type=int, default=int(os.environ.get("LANGUAGE_SHARD_INDEX", "0")))
    parser.add_argument(
        "--build-views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build label-blind candidate views and separate correctness statistics after pooling.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=None,
        help="Override manifest candidate_views (for example: all_usable random_4 diverse_4).",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Run regimes without pooling or derived views; used by parallel cluster workers.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    ns = parse_args()
    manifest = load_manifest(ns.manifest)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    (ns.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_dirs = []
    for regime in manifest["regimes"]:
        run_dir = ns.output_dir / "runs" / regime["name"]
        run_generation(ns, manifest, regime, run_dir)
        run_dirs.append(run_dir)

    if ns.generate_only:
        return

    pooled_dir = ns.output_dir / "pooled"
    run_merge(pooled_dir, run_dirs, ns.overwrite)
    if ns.build_views:
        run_build_views(ns.output_dir, pooled_dir, manifest, ns.views, ns.overwrite)


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("regimes"), list) or not data["regimes"]:
        raise ValueError(f"Manifest must contain a non-empty regimes list: {path}")
    names = [regime.get("name") for regime in data["regimes"]]
    if len(names) != len(set(names)):
        raise ValueError("Regime names must be unique.")
    return data


def run_generation(ns: argparse.Namespace, manifest: dict[str, Any], regime: dict[str, Any], run_dir: Path) -> None:
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
        str(regime.get("candidate_source", defaults.get("candidate_source", "main"))),
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
        str(regime.get("n", defaults.get("n", 1))),
        "--batch-size",
        str(regime.get("batch_size", defaults.get("batch_size", 128))),
        "--seed",
        str(regime.get("seed", defaults.get("seed", 0))),
        "--tolerance",
        str(regime.get("tolerance", defaults.get("tolerance", "0"))),
        "--language-shard-count",
        str(ns.language_shard_count),
        "--language-shard-index",
        str(ns.language_shard_index),
    ]
    append_optional_flags(cmd, regime, defaults)
    if ns.languages:
        cmd.extend(["--languages", *ns.languages])
    if ns.limit_per_language is not None:
        cmd.extend(["--limit-per-language", str(ns.limit_per_language)])
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
    if not regime.get("use_stop_sequences", defaults.get("use_stop_sequences", True)):
        cmd.append("--no-use-stop-sequences")
    preamble_file = regime.get("assistant_preamble_file", defaults.get("assistant_preamble_file"))
    if preamble_file:
        cmd.extend(["--assistant-preamble-file", str(preamble_file)])


def run_merge(pooled_dir: Path, run_dirs: list[Path], overwrite: bool) -> None:
    cmd = [sys.executable, "scripts/merge_candidate_pools.py", "--output-dir", str(pooled_dir)]
    for run_dir in run_dirs:
        cmd.extend(["--source-dir", f"{run_dir.name}={run_dir}"])
    if overwrite:
        cmd.append("--overwrite")
    print_command(cmd)
    subprocess.run(cmd, check=True)


def run_build_views(
    output_dir: Path,
    pooled_dir: Path,
    manifest: dict[str, Any],
    views_override: list[str] | None,
    overwrite: bool,
) -> None:
    views = views_override or manifest.get(
        "candidate_views", ["all_usable", "random_4", "diverse_4"]
    )
    cmd = [
        sys.executable,
        "scripts/build_candidate_views.py",
        "--input-dir",
        str(pooled_dir),
        "--output-dir",
        str(output_dir),
        "--views",
        *[str(view) for view in views],
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


def print_command(cmd: list[str]) -> None:
    print("[run_experiment] Running:", " ".join(shell_quote(part) for part in cmd), flush=True)


def shell_quote(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return repr(value)
    return value


if __name__ == "__main__":
    main()
