# Experiment 2: all-67 fine-tuning ablation

## Why

The main fine-tuning experiments train on 24 core languages + English, chosen by a pre-registered rule. A reviewer may ask whether training on every language would change the conclusions. This ablation trains the same student with the same recipe on all 67 languages and compares the two.

The main comparison is the gap between two arms, `method` − `translated`:

| Arm | Trains on |
|---|---|
| `translated` | the dataset's translated gold solutions |
| `method` | the model's own reasoning chains, labelled by our label-free method |

The ablation passes if that gap has the same sign here as in the 24-language runs.

## What runs

| Setting | Value |
|---|---|
| Student | Qwen3-14B-Base, LoRA r=16 / alpha=32 on all linear layers, lr 2e-4, 1 epoch, max length 1,024 tokens, seed 1 |
| `translated` arm | `data/all67_ablation/sft_translated.jsonl`: 500,691 examples (7,473 problems × 67 languages) |
| `method` arm | `data/all67_ablation/sft_method.jsonl`: arrives later, after the 67-language labelling on the original cluster |
| Tests (base model + adapter) | `test_core25_languages`: full GSM8K test × 24 core languages + English |
| | `test_42_extension_languages`: full GSM8K test × the other 42 languages |
| | `test_mgsm_human_translations`: MGSM's human translations |

Test problems 0 and 5 are removed from every test set: they are the generation prompt's examples.

## Steps

| Step | Command | Pass looks like |
|---|---|---|
| 0. Smoke test (~1 h) | `bash 0_smoke_test.sh` | both jobs end; `results/all67_ablation/*__smoke.json` exist. The training log shows `500691 examples, N dropped (> 1024 tokens ...)` and a 20-step progress bar. |
| 1. `translated` arm | `bash 1_submit_translated.sh` | `results/all67_ablation/<test set>__translated_s1.json` for all three tests |
| 2. `method` arm (later) | `bash 2_submit_method.sh` | the same files with `__method_s1` |

Send back `results/all67_ablation/` and the `all67-train-*` logs; the logs record how many examples were dropped as too long.

## Files

| File | Role |
|---|---|
| `0_smoke_test.sh`, `1_submit_translated.sh`, `2_submit_method.sh` | submit the jobs (read `config.env`) |
| `train.sbatch` | LoRA training on 4 GPUs (the project's `train_lora.py`, unchanged); adapter goes to `$OUTPUT_DIR/all67_ablation/adapters/<arm>_s1` |
| `evaluate.sbatch` | greedy evaluation of the base model + adapters on the three tests (the project's `evaluate.py`, unchanged) |
