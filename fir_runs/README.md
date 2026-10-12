# Remaining core-24 trainings and evaluations on Fir

Runs all the core-24 fine-tuning that hadn't finished on Rorqual, with the Rorqual job scripts unchanged: 14 trainings (2 GPUs each) and 4 evaluations (1 GPU each). The recipe (LoRA r16/a32, lr 2e-4, 1 epoch, max length 1,024, effective batch 32, seeds) lives inside `jobs/core24_train.sbatch`, a byte-identical copy of Rorqual's.

| Student | Trainings | Evaluation (base model + these adapters; core test, 42 extension languages, MGSM human) |
|---|---|---|
| Mistral-Nemo-Base-2407 | gold_filtered s2; method s1, s2; method_lenmatch s1 | one job, tag `fir` |
| Qwen3-14B-Base | method s1, s2; method_lenmatch s1 | tag `wave3` |
| gemma-3-12b-pt-text | own_unlabelled s1, s2; gold_filtered s1, s2 | tag `wave2` |
| gemma-3-12b-pt-text | method s1, s2; method_lenmatch s1 | tag `wave3` |

## Steps (after `fir_labelling/setup_fir.sh` has run)

1. **Hugging Face login.** Mistral and Gemma are gated, so log in with an account that has accepted both licences:
   ```
   HF_HOME=/scratch/$USER/cache/hf ~/venvs/multilingual-icm/bin/hf auth login
   ```
2. **Setup** (login node): `bash ~/multilingual_icm/ICM_work_/fir_runs/setup_runs.sh`. It:
   - builds the training environment with `uv` (torch 2.13.0, transformers 4.57.6, peft 0.19.1, accelerate 1.14.0);
   - unpacks and verifies the 7 training and test files;
   - downloads Mistral and Gemma at the exact revisions Rorqual used.
3. **Dry run:** `bash ~/multilingual_icm/ICM_work_/fir_runs/submit_runs.sh`. If a request doesn't fit Fir's nodes, adjust TRAIN_CPUS, TRAIN_MEM, EVAL_CPUS or EVAL_MEM. These never change results.
4. **Submit:** `SUBMIT=1 bash ~/multilingual_icm/ICM_work_/fir_runs/submit_runs.sh`. The chain is prepare → 14 trainings → 4 evaluations.
   - The prepare job checks Mistral, Gemma and the extracted text-only Gemma against Rorqual's checksums.
   - Any mismatch stops everything.

## Checks

- **Each training log:** `[train] 2 GPU(s), grad-accum 4, effective batch 32`, plus the `examples, N dropped` line.
- **Each evaluation log:** `Using FlashAttention version 3`.
- **After the first evaluation:** run `python ~/multilingual_icm/ICM_work_/fir_runs/check_stack.py`. Every evaluation also scores the base model, so Fir's base accuracy must equal Rorqual's (in `reference/`), language by language, within 0.5 points. That shows the two clusters' results are directly comparable.

## Results

They stay on Fir until they're brought back:

| What | Where |
|---|---|
| Accuracy files | `~/multilingual_icm/data/findings/results/core24_*_{fir,wave2,wave3}.json` |
| Job logs | `~/multilingual_icm/data/findings/logs/` |
| Adapters | `/scratch/$USER/multilingual-icm/results/core24/` |

Report the file list, the `check_stack.py` output and each job's runtime and GPU type.
