# q09: does fine-tuning on the method's data beat fine-tuning on translated data?

**Pilot on the existing 37 languages × 1,500 problems** before the 50-language run.
Qwen3-14B-Base (the model that generated and labelled the data) is fine-tuned
with LoRA on each of two training sets and evaluated on the held-out test split.

| arm | completion for each (language, problem) | gold used? |
|---|---|---|
| `translated` | the dataset's translated gold solution | yes, by construction |
| `method` | one of the model's own chains agreeing with the answer the full pipeline committed (vote + resampling + problem-disjoint vote × probe fusion), picked at random | only the 300-problem calibration budget, never the problem's own gold |
| `own_unlabelled` | one of the model's own chains, picked at random, **no labelling** (self-distillation control) | no |
| `method_lenmatch` | the method's label, the chain closest in length to the translated solution (length control) | as `method` |

Seeds: `translated`, `method`, `own_unlabelled` × 2; `method_lenmatch` × 1. Arms
are compared seed-averaged with a paired bootstrap; the seed-to-seed
difference is reported as the training-noise yardstick.

**Matched on everything except the solution text:** same model, same 37
languages, the same (language, problem) pairs in both arms (one example each,
so counts match per language), same prompt (`Question: … / Answer:`), same
completion cut-off (up to the `#### N` line), same LoRA recipe and seed, same
greedy evaluation with the same answer parser and exact match. The builder
prints per-language lengths of both arms and flags any language where they
differ by more than a third. Gold is read only after the sets are written, to
report how many method examples are actually correct.

Two method arms: `method` (voting + regeneration) and `method_full` (the
same, plus the probe channel: high-resource pools the vote marked contested
are labelled by a 4-fold probe never trained on that language; see
probe_work_ `q09_probe_labels_for_finetuning`). They differ only on the pools
the probe relabelled, so their comparison isolates the probe channel.

| file | step |
|---|---|
| `build_sft_sets.py` / `.sbatch` | CPU: the three training sets + the test-split eval set (250 problems × 37 languages) |
| `train_lora.py` / `.sbatch` | GPU 4 × H100: LoRA r=16 on all linear layers, 1 epoch, lr 2e-4; `ARM=translated`, `method` or `method_full` |
| `evaluate.py` / `.sbatch` | GPU 1 × H100: base and the three arms; accuracy overall, by tier, by language |

Outputs: `data/ICM_work_/runs/q09_*.jsonl`, `q09_results.json`; adapters in
`/scratch/.../results/q09/adapters/`.
