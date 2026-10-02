# q09: does fine-tuning on the method's data beat fine-tuning on translated data?

**Pilot on the existing 37 languages × 1,500 problems** before the 50-language run.
Qwen3-14B-Base (the model that generated and labelled the data) is fine-tuned
with LoRA on each of two training sets and evaluated on the held-out test split.

| arm | completion for each (language, problem) | gold used? |
|---|---|---|
| `translated` | the dataset's translated gold solution | yes, by construction |
| `method` | one of the model's own chains agreeing with the answer the method committed (voting + regeneration, record rules), picked at random | **no** |

**Matched on everything except the solution text:** same model, same 37
languages, the same (language, problem) pairs in both arms (one example each,
so counts match per language), same prompt (`Question: … / Answer:`), same
completion cut-off (up to the `#### N` line), same LoRA recipe and seed, same
greedy evaluation with the same answer parser and exact match. The builder
prints per-language lengths of both arms and flags any language where they
differ by more than a third. Gold is read only after the sets are written, to
report how many method examples are actually correct.

Not in this pilot: the probe channel for high-resource hard pools (it needs
activations for every chain, which exist only for a subset); the method arm
here is the voting + regeneration labels.

| file | step |
|---|---|
| `build_sft_sets.py` / `.sbatch` | CPU: both training sets + the test-split eval set (250 problems × 37 languages) |
| `train_lora.py` / `.sbatch` | GPU 4 × H100: LoRA r=16 on all linear layers, 1 epoch, lr 2e-4; `ARM=translated` or `ARM=method` |
| `evaluate.py` / `.sbatch` | GPU 1 × H100: base, translated, method; accuracy overall, by tier, by language |

Outputs: `data/ICM_work_/runs/q09_*.jsonl`, `q09_results.json`; adapters in
`/scratch/.../results/q09/adapters/`.
