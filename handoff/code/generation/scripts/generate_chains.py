from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi
from rich.console import Console
from tqdm import tqdm
from vllm import LLM, SamplingParams

from answer_parsing import extract_gold_answer, extract_prediction_answer, numbers_match
from dataset_loading import discover_language_datasets, get_question, load_language_dataset
from prompts import (
    DEFAULT_FEW_SHOT_ORIGINAL_IDS,
    NATIVE_FEW_SHOT_COUNT,
    SUPPORTED_FEW_SHOT_COUNTS,
    available_prompt_names,
    derive_stop_sequences,
    get_prompt_language,
    render_prompt,
)


console = Console()


@dataclass(frozen=True)
class GenerateArgs:
    model: str
    dataset: str
    split: str
    output_dir: Path
    regime_name: str
    candidate_source: str
    prompt_template: str
    english_prompt_only: bool
    assistant_preamble_file: Path | None
    few_shot_count: int
    few_shot_split: str
    few_shot_original_ids: tuple[int, ...]
    dtype: str
    quantization: str | None
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    enable_prefix_caching: bool
    max_model_len: int
    max_new_tokens: int
    temperature: float
    top_p: float
    n: int
    use_stop_sequences: bool
    batch_size: int
    seed: int
    tolerance: Decimal
    languages: list[str] | None
    id_filter_dir: Path | None
    language_shard_count: int
    language_shard_index: int
    limit_per_language: int | None
    overwrite: bool
    trust_remote_code: bool
    hf_results_repo: str | None
    hf_results_path: str | None
    hf_private: bool
    hf_upload_each_language: bool


def parse_args() -> GenerateArgs:
    parser = argparse.ArgumentParser(description="Generate multilingual GSM8K candidate chains.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--dataset", default="open-multilang/gsm8k-106")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_8b_base_train_chains"))
    parser.add_argument("--regime-name", default="default")
    parser.add_argument(
        "--candidate-source",
        default="main",
        choices=["main", "retry", "wrong_chain_ablation"],
        help="Provenance label retained during pooling, view construction, and reporting.",
    )
    parser.add_argument(
        "--prompt-template",
        default="zero_shot_cot",
        help=f"Prompt template name. Known now: {', '.join(available_prompt_names())}",
    )
    parser.add_argument("--english-prompt-only", action="store_true")
    parser.add_argument(
        "--assistant-preamble-file",
        type=Path,
        default=None,
        help="Optional preamble text, e.g. a short HHH-style preamble for base models.",
    )
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=0,
        help="Number of fixed test-split native demonstrations. few_shot_cot requires 6.",
    )
    parser.add_argument("--few-shot-split", default="test")
    parser.add_argument(
        "--few-shot-original-ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_FEW_SHOT_ORIGINAL_IDS),
        help="Fixed original_id values selected from --few-shot-split for every language.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=12, help="Completions to sample per question.")
    parser.add_argument(
        "--use-stop-sequences",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop generation at template-derived markers (native question label, <example>) "
        "so base-model samples end after the answer instead of echoing the prompt.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", default="0")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument(
        "--id-filter-dir",
        type=Path,
        default=None,
        help="Optional directory with <language>.txt files listing example ids to generate.",
    )
    parser.add_argument("--language-shard-count", type=int, default=1)
    parser.add_argument("--language-shard-index", type=int, default=0)
    parser.add_argument("--limit-per-language", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-results-repo", default=None)
    parser.add_argument("--hf-results-path", default=None)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-upload-each-language", action=argparse.BooleanOptionalAction, default=True)
    ns = parser.parse_args()

    if ns.few_shot_count < 0:
        parser.error("--few-shot-count must be >= 0")
    if ns.prompt_template == "few_shot_cot" and ns.few_shot_count not in SUPPORTED_FEW_SHOT_COUNTS:
        parser.error(
            f"few_shot_cot requires --few-shot-count in {SUPPORTED_FEW_SHOT_COUNTS}"
        )
    if ns.prompt_template == "few_shot_cot_2shot" and ns.few_shot_count != 2:
        parser.error(
            "few_shot_cot_2shot headers say 'two examples'; --few-shot-count must be 2"
        )
    if ns.few_shot_count > len(ns.few_shot_original_ids):
        parser.error("--few-shot-count exceeds the number of --few-shot-original-ids")
    if ns.n < 1:
        parser.error("--n must be >= 1")
    if ns.language_shard_count < 1:
        parser.error("--language-shard-count must be >= 1")
    if not 0 <= ns.language_shard_index < ns.language_shard_count:
        parser.error("--language-shard-index must satisfy 0 <= index < count")

    return GenerateArgs(
        model=ns.model,
        dataset=ns.dataset,
        split=ns.split,
        output_dir=ns.output_dir,
        regime_name=ns.regime_name,
        candidate_source=ns.candidate_source,
        prompt_template=ns.prompt_template,
        english_prompt_only=ns.english_prompt_only,
        assistant_preamble_file=ns.assistant_preamble_file,
        few_shot_count=ns.few_shot_count,
        few_shot_split=ns.few_shot_split,
        few_shot_original_ids=tuple(ns.few_shot_original_ids),
        dtype=ns.dtype,
        quantization=ns.quantization,
        tensor_parallel_size=ns.tensor_parallel_size,
        gpu_memory_utilization=ns.gpu_memory_utilization,
        max_num_seqs=ns.max_num_seqs,
        max_num_batched_tokens=ns.max_num_batched_tokens,
        enable_prefix_caching=ns.enable_prefix_caching,
        max_model_len=ns.max_model_len,
        max_new_tokens=ns.max_new_tokens,
        temperature=ns.temperature,
        top_p=ns.top_p,
        n=ns.n,
        use_stop_sequences=ns.use_stop_sequences,
        batch_size=ns.batch_size,
        seed=ns.seed,
        tolerance=Decimal(ns.tolerance),
        languages=ns.languages,
        id_filter_dir=ns.id_filter_dir,
        language_shard_count=ns.language_shard_count,
        language_shard_index=ns.language_shard_index,
        limit_per_language=ns.limit_per_language,
        overwrite=ns.overwrite,
        trust_remote_code=ns.trust_remote_code,
        hf_results_repo=ns.hf_results_repo,
        hf_results_path=ns.hf_results_path,
        hf_private=ns.hf_private,
        hf_upload_each_language=ns.hf_upload_each_language,
    )


def main() -> None:
    args = parse_args()
    prepare_output_dir(args)
    assistant_preamble = load_optional_text(args.assistant_preamble_file)

    console.print(f"[bold]Loading dataset[/bold] {args.dataset} ({args.split})")
    language_datasets = discover_language_datasets(args.dataset, args.split, languages=args.languages)
    language_datasets = shard_language_datasets(
        language_datasets,
        shard_count=args.language_shard_count,
        shard_index=args.language_shard_index,
    )
    console.print(f"Discovered {len(language_datasets)} language datasets.")

    console.print(f"[bold]Loading model[/bold] {args.model}")
    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)

    summaries: list[dict[str, Any]] = []
    for item in language_datasets:
        stop_sequences = build_stop_sequences(args, item.language)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.n,
            seed=args.seed,
            stop=stop_sequences or None,
        )
        if stop_sequences:
            console.print(f"Stop sequences for {item.language}: {stop_sequences!r}")
        demonstrations = load_native_demonstrations(args, item.language)
        summary = generate_language(
            args,
            llm,
            sampling_params,
            item.language,
            item.dataset,
            assistant_preamble,
            demonstrations,
        )
        summaries.append(summary)
        write_summary(args.output_dir, summaries)
        if args.hf_results_repo and args.hf_upload_each_language:
            upload_results_to_hf(args, f"Upload {item.language} generated chains")

    if args.hf_results_repo:
        upload_results_to_hf(args, "Upload completed generated chains")
    console.print(f"[green]Done.[/green] Summary: {args.output_dir / 'summary' / 'generation_by_language.csv'}")


def build_stop_sequences(args: GenerateArgs, language: str) -> list[str]:
    if not args.use_stop_sequences:
        return []
    return derive_stop_sequences(
        args.prompt_template,
        language=language,
        english_prompt_only=args.english_prompt_only,
    )


def prepare_output_dir(args: GenerateArgs) -> None:
    (args.output_dir / "generated").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary").mkdir(parents=True, exist_ok=True)
    run_config = asdict(args)
    run_config["output_dir"] = str(args.output_dir)
    run_config["id_filter_dir"] = str(args.id_filter_dir) if args.id_filter_dir else None
    run_config["assistant_preamble_file"] = (
        str(args.assistant_preamble_file) if args.assistant_preamble_file else None
    )
    run_config["tolerance"] = str(args.tolerance)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def shard_language_datasets(language_datasets: list[Any], shard_count: int, shard_index: int) -> list[Any]:
    if shard_count == 1:
        return language_datasets
    return [item for index, item in enumerate(language_datasets) if index % shard_count == shard_index]


def generate_language(
    args: GenerateArgs,
    llm: LLM,
    sampling_params: SamplingParams,
    language: str,
    dataset: Any,
    assistant_preamble: str,
    demonstrations: list[dict[str, str]],
) -> dict[str, Any]:
    output_path = args.output_dir / "generated" / f"{safe_filename(language)}.jsonl"
    if args.limit_per_language is not None:
        dataset = dataset.select(range(min(args.limit_per_language, len(dataset))))
    dataset = filter_dataset_by_ids(dataset, language, args.id_filter_dir)

    expected = len(dataset)
    if output_path.exists() and not args.overwrite:
        existing_rows = count_lines(output_path)
        if existing_rows == expected:
            console.print(f"[cyan]Skipping[/cyan] {language}: {existing_rows}/{expected} rows already exist.")
            return summarize_generated_file(language, output_path)
        console.print(f"[yellow]Rewriting[/yellow] {language}: found {existing_rows}, expected {expected}.")

    tokenizer = llm.get_tokenizer()
    max_prompt_tokens = args.max_model_len - args.max_new_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("--max-new-tokens must be smaller than --max-model-len.")

    rows = build_prompt_rows(
        args,
        dataset,
        language,
        tokenizer,
        max_prompt_tokens,
        assistant_preamble,
        demonstrations,
    )
    overlong_count = sum(1 for row in rows if row["prompt_too_long"])
    console.print(
        f"[bold]Generating[/bold] {language} / {args.regime_name} "
        f"({len(rows)} examples x {args.n} chains)"
    )
    if overlong_count:
        console.print(f"[yellow]Skipping[/yellow] {language}: {overlong_count} overlong prompts.")

    with output_path.open("w", encoding="utf-8") as handle:
        for start in tqdm(range(0, len(rows), args.batch_size), desc=language, leave=False):
            batch = rows[start : start + args.batch_size]
            runnable_rows = []
            for row in batch:
                if row["prompt_too_long"]:
                    handle.write(json.dumps({**row, "candidates": [], "skipped": True}, ensure_ascii=False) + "\n")
                else:
                    runnable_rows.append(row)
            handle.flush()

            if not runnable_rows:
                continue

            outputs = llm.generate([row["prompt"] for row in runnable_rows], sampling_params, use_tqdm=False)
            for row, output in zip(runnable_rows, outputs, strict=True):
                gold_number = Decimal(row["gold_number"]) if row["gold_number"] is not None else None
                candidates = []
                for sample_index, sample in enumerate(output.outputs):
                    generated = sample.text
                    pred_raw, pred_number = extract_prediction_answer(generated)
                    candidates.append(
                        {
                            "regime_name": args.regime_name,
                            "candidate_source": args.candidate_source,
                            "prompt_template": args.prompt_template,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "seed": args.seed,
                            "n": args.n,
                            "prompt_language": row["prompt_language"],
                            "few_shot_count": args.few_shot_count,
                            "sample_index": sample_index,
                            "text": generated,
                            "finish_reason": getattr(sample, "finish_reason", None),
                            "stop_reason": getattr(sample, "stop_reason", None),
                            "cumulative_logprob": getattr(sample, "cumulative_logprob", None),
                            "pred_raw": pred_raw,
                            "pred_number": str(pred_number) if pred_number is not None else None,
                            "correct": numbers_match(pred_number, gold_number, args.tolerance),
                        }
                    )
                handle.write(
                    json.dumps({**row, "candidates": candidates, "skipped": False}, ensure_ascii=False) + "\n"
                )
            handle.flush()

    return summarize_generated_file(language, output_path)


def build_prompt_rows(
    args: GenerateArgs,
    dataset: Any,
    language: str,
    tokenizer: Any,
    max_prompt_tokens: int,
    assistant_preamble: str,
    demonstrations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    columns = dataset.column_names
    rows = []
    for idx, example in enumerate(dataset):
        question = get_question(example, columns)
        gold_raw, gold_number = extract_gold_answer(example)
        prompt = render_prompt(
            question,
            args.prompt_template,
            language=language,
            english_prompt_only=args.english_prompt_only,
            assistant_preamble=assistant_preamble,
            demonstrations=demonstrations or None,
        )
        prompt_language = get_prompt_language(
            args.prompt_template,
            language=language,
            english_prompt_only=args.english_prompt_only,
        )
        prompt_token_count = count_prompt_tokens(tokenizer, prompt)
        rows.append(
            {
                "id": str(example.get("id", idx)),
                "original_id": example.get("original_id"),
                "split": example.get("split"),
                "language": language,
                "language_name": example.get("language_name"),
                "regime_name": args.regime_name,
                "candidate_source": args.candidate_source,
                "prompt_template": args.prompt_template,
                "prompt_language": prompt_language,
                "few_shot_count": args.few_shot_count,
                "few_shot_split": args.few_shot_split if demonstrations else None,
                "few_shot_original_ids": (
                    list(args.few_shot_original_ids[: args.few_shot_count]) if demonstrations else []
                ),
                "question": question,
                "original_question": example.get("original_question"),
                "gold_raw": gold_raw,
                "gold_number": str(gold_number) if gold_number is not None else None,
                "original_answer": example.get("original_answer"),
                "original_answer_number": example.get("original_answer_number"),
                "prompt": prompt,
                "prompt_token_count": prompt_token_count,
                "max_prompt_tokens": max_prompt_tokens,
                "prompt_too_long": prompt_token_count > max_prompt_tokens,
            }
        )
    return rows


def load_native_demonstrations(args: GenerateArgs, language: str) -> list[dict[str, str]]:
    if args.few_shot_count == 0:
        return []

    dataset = load_language_dataset(args.dataset, args.few_shot_split, language)
    wanted_ids = list(args.few_shot_original_ids[: args.few_shot_count])
    by_original_id = {int(example["original_id"]): example for example in dataset}
    missing = [original_id for original_id in wanted_ids if original_id not in by_original_id]
    if missing:
        raise ValueError(
            f"Missing few-shot original_id values for {language}/{args.few_shot_split}: {missing}"
        )

    demonstrations = []
    for original_id in wanted_ids:
        example = by_original_id[original_id]
        answer = str(example.get("answer") or "").strip()
        if "####" not in answer:
            _, gold_number = extract_gold_answer(example)
            answer = f"{answer}\n#### {gold_number}".strip()
        demonstrations.append(
            {
                "original_id": str(original_id),
                "question": str(example["question"]),
                "answer": answer,
            }
        )
    return demonstrations


def summarize_generated_file(language: str, path: Path) -> dict[str, Any]:
    total = 0
    skipped = 0
    candidates = 0
    parseable = 0
    correct = 0
    by_source: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            skipped += int(bool(record.get("skipped")))
            for candidate in record.get("candidates", []):
                candidates += 1
                source = str(candidate.get("candidate_source") or record.get("candidate_source") or "unknown")
                by_source[source] = by_source.get(source, 0) + 1
                parseable += int(candidate.get("pred_number") is not None)
                correct += int(bool(candidate.get("correct")))
    return {
        "language": language,
        "total": total,
        "skipped": skipped,
        "candidates": candidates,
        "parseable_candidates": parseable,
        "correct_candidates": correct,
        "candidates_by_source": by_source,
        "candidate_file": str(path),
    }


def write_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    summary_dir = output_dir / "summary"
    sorted_summaries = sorted(summaries, key=lambda row: row["language"])
    frame = pd.DataFrame(sorted_summaries)
    frame.to_csv(summary_dir / "generation_by_language.csv", index=False)
    (summary_dir / "generation_by_language.json").write_text(
        json.dumps(sorted_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upload_results_to_hf(args: GenerateArgs, commit_message: str) -> None:
    repo_path = args.hf_results_path or args.output_dir.name
    console.print(f"[bold]Uploading results[/bold] to {args.hf_results_repo}/{repo_path}")
    api = HfApi()
    for attempt in range(1, 4):
        try:
            api.create_repo(args.hf_results_repo, repo_type="dataset", private=args.hf_private, exist_ok=True)
            api.upload_folder(
                repo_id=args.hf_results_repo,
                repo_type="dataset",
                folder_path=str(args.output_dir),
                path_in_repo=repo_path,
                commit_message=commit_message,
            )
            return
        except Exception as exc:
            if attempt == 3:
                write_upload_failure(args.output_dir, repo_path, commit_message, exc)
                return
            time.sleep(attempt * 10)


def write_upload_failure(output_dir: Path, repo_path: str, commit_message: str, exc: Exception) -> None:
    failure_path = output_dir / "hf_upload_failures.jsonl"
    record = {
        "repo_path": repo_path,
        "commit_message": commit_message,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    with failure_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_optional_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8").strip()


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    try:
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(prompt))


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def filter_dataset_by_ids(dataset: Any, language: str, id_filter_dir: Path | None) -> Any:
    if id_filter_dir is None:
        return dataset
    filter_path = id_filter_dir / f"{safe_filename(language)}.txt"
    if not filter_path.exists():
        return dataset.select([])
    wanted = {
        line.split("#", 1)[0].strip()
        for line in filter_path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }
    if not wanted:
        return dataset.select([])
    indices = [idx for idx, example in enumerate(dataset) if str(example.get("id", idx)) in wanted]
    return dataset.select(indices)


if __name__ == "__main__":
    main()
