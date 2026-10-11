from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from answer_parsing import extract_gold_answer
from dataset_loading import load_language_dataset
from prompts import DEFAULT_FEW_SHOT_ORIGINAL_IDS, render_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate offline assets and native few-shot prompts.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--language", default="af")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    train = load_language_dataset(str(args.dataset_dir), "train", args.language)
    test = load_language_dataset(str(args.dataset_dir), "test", args.language)
    by_original_id = {int(example["original_id"]): example for example in test}

    demonstrations = []
    for original_id in DEFAULT_FEW_SHOT_ORIGINAL_IDS:
        if original_id not in by_original_id:
            raise ValueError(f"Missing test original_id={original_id} for {args.language}")
        example = by_original_id[original_id]
        answer = str(example["answer"]).strip()
        if "####" not in answer:
            _, gold_number = extract_gold_answer(example)
            answer = f"{answer}\n#### {gold_number}"
        demonstrations.append({"question": str(example["question"]), "answer": answer})

    prompt = render_prompt(
        str(train[0]["question"]),
        "few_shot_cot",
        language=args.language,
        demonstrations=demonstrations,
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    max_prompt_tokens = args.max_model_len - args.max_new_tokens
    if prompt_tokens > max_prompt_tokens:
        raise ValueError(
            f"Prompt has {prompt_tokens} tokens, exceeding the {max_prompt_tokens}-token budget."
        )

    print(f"model={args.model_dir}")
    print(f"dataset={args.dataset_dir}")
    print(f"language={args.language} train_rows={len(train)} test_rows={len(test)}")
    print(f"few_shot_original_ids={list(DEFAULT_FEW_SHOT_ORIGINAL_IDS)}")
    print(f"prompt_tokens={prompt_tokens} max_prompt_tokens={max_prompt_tokens}")
    print(f"cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}")
    print("preflight=ok")


if __name__ == "__main__":
    main()
