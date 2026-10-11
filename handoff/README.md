# Handoff: Gemma generation + all-67 fine-tuning ablation

Two self-contained experiments from the "Wrong in Every Language" project. They run on any Slurm cluster with 80 GB GPUs (H100/A100). This folder holds all the code; the data is rebuilt from pinned public sources by one script, and the model weights are downloaded.

| # | Experiment | What it produces | Compute |
|---|---|---|---|
| 1 | [Gemma generation](experiments/1_gemma_generation/README.md) | 20 reasoning chains per problem from Gemma 3 12B (base), for 24 core languages + English, on all 7,473 GSM8K training problems | 4 jobs × 4 GPUs × ~4–7 h, plus 1 CPU merge |
| 2 | [All-67 fine-tuning ablation](experiments/2_all67_finetuning_ablation/README.md) | Qwen3 14B base fine-tuned on all 67 languages (the main runs use 24) and its test accuracy | 1 job × 4 GPUs × ~8 h, plus 1 GPU × ~3 h for evaluation; repeated later for a second arm |

The two experiments are independent; run them in either order or in parallel.

## Steps

1. **Settings.** Edit `config.env`, the only file you edit: accounts, storage paths, Python environments, GPU type.
2. **Environments.** Build the two Python environments: see [setup/README.md](setup/README.md).
3. **Data.** Git holds no data. Rebuild it from pinned public sources; the build verifies every table and file against the original cluster and stops on any difference:
   ```
   source ./config.env && source $VENV_VLLM/bin/activate
   python setup/build_data.py download     # login node (needs internet), ~600 MB
   sbatch --account=$ACCOUNT_CPU setup/build_data.sbatch   # offline, ~1 h; its log ends "DATA BUILT AND VERIFIED"
   ```
   On a cluster with environment modules, load `arrow` first if `datasets` needs it (Alliance clusters: `module load python/3.11 arrow`).
4. **Models.** Run `bash setup/download_models.sh` (~40 GB; Gemma needs a Hugging Face login).
5. **Check.** Run `bash setup/check_setup.sh` and continue only when it prints `ALL CHECKS PASSED`.
6. **Smoke tests.** Under 1 h each:
   ```
   bash experiments/1_gemma_generation/0_smoke_test.sh
   bash experiments/2_all67_finetuning_ablation/0_smoke_test.sh
   ```
   Each experiment's README says what a pass looks like.
7. **Real runs:**
   ```
   bash experiments/1_gemma_generation/1_submit_generation.sh
   bash experiments/2_all67_finetuning_ablation/1_submit_translated.sh
   ```
8. **Later:** `experiments/2_all67_finetuning_ablation/2_submit_method.sh`, once you receive `sft_method.jsonl`.

Job logs go to `experiments/<experiment>/logs/`.

## What to send back

| Experiment | Send |
|---|---|
| 1 | `$OUTPUT_DIR/gemma_generation/final/`: the merged chains, about 1–2 GB |
| 1 | the 4 chunk logs and the merge log from `experiments/1_gemma_generation/logs/` |
| 2 | everything in `results/all67_ablation/`: small JSON files of accuracies |
| 2 | the training logs, which record how many examples were dropped as too long |

Do not send the adapters.

## Ground rules: these keep the results comparable with the original cluster

- **Change no settings in the code or data.** That covers sampling, LoRA recipe, seeds, prompts and test sets. Only `config.env` is meant to be edited.
- **Use the pinned library versions** in `setup/requirements-*.txt`. Other versions of vLLM or transformers can shift results slightly.
- **If something fails, send the log rather than patching around it.** A silent workaround can make results incomparable.

## Folder map

See [FILES.md](FILES.md) for every file, what it does and where it came from.
