"""Q09 step 2 (GPU): LoRA fine-tuning of Qwen3-14B-Base on one arm's training set.

Run once per arm with IDENTICAL arguments except --data/--out (the .sbatch
files do exactly that). Loss on the completion tokens only. Examples longer
than --max-len tokens are dropped and the count is printed; both arms use the
same limit.

    torchrun --nproc_per_node 4 -m icm.experiments.q09_does_the_dataset_beat_translation.train_lora \
        --data <q09_sft_*.jsonl> --out <adapter dir>
"""
import argparse
import json
import os

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N steps")
    ap.add_argument("--match", action="append", default=[],
                    help="other arms' files: an id too long in ANY arm is dropped from this one too, "
                         "so every arm trains on exactly the same pairs")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)
    set_seed(a.seed)
    rank0 = int(os.environ.get("RANK", "0")) == 0

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def too_long(path):
        out = set()
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            n = len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) + \
                len(tok(r["completion"], add_special_tokens=False)["input_ids"]) + 1
            if n > a.max_len:
                out.add(r["id"])
        return out

    excluded = set().union(*[too_long(f) for f in [a.data] + a.match])
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    examples, dropped = [], 0
    for r in rows:
        if r["id"] in excluded:
            dropped += 1
            continue
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(r["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        examples.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    if rank0:
        n_tok = sum(len(e["input_ids"]) for e in examples)
        n_sup = sum(sum(1 for t in e["labels"] if t != -100) for e in examples)
        print(f"{a.data}: {len(rows)} examples, {dropped} dropped (> {a.max_len} tokens in this or a matched arm), "
              f"{len(examples)} kept; {n_tok} tokens, {n_sup} supervised", flush=True)

    def collate(batch):
        n = max(len(e["input_ids"]) for e in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, e in enumerate(batch):
            L = len(e["input_ids"])
            ids[i, :L] = torch.tensor(e["input_ids"]); lab[i, :L] = torch.tensor(e["labels"]); att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if rank0:
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.per_device_batch, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        bf16=True, logging_steps=20, save_strategy="no", report_to=[], seed=a.seed,
        data_seed=a.seed, ddp_find_unused_parameters=False, remove_unused_columns=False,
        dataloader_num_workers=2, group_by_length=False)
    Trainer(model=model, args=args, train_dataset=examples, data_collator=collate).train()
    if rank0:
        model.save_pretrained(a.out)
        json.dump(vars(a), open(os.path.join(a.out, "q09_train_args.json"), "w"), indent=1)
        print(f"saved adapter to {a.out}")


if __name__ == "__main__":
    main()
