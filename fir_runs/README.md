# Remaining core-24 fine-tuning on Fir and Nibi

All trainings and evaluations still to do for the paper's fine-tuning results. They moved from Rorqual (whose queue
stalled) to Fir. The job scripts in `jobs/` are byte-identical copies of Rorqual's `data/findings/jobs/`:
same recipe, seeds and test sets, with absolute paths that exist on Fir under the same username.

## Splitting across clusters
Set `STUDENTS` the same way for setup and submit on each cluster, so each one runs only its own students (all of a student's runs stay on one cluster):
- Fir: `STUDENTS="Qwen3-14B-Base Mistral-Nemo-Base-2407"`
- Nibi: `STUDENTS="gemma-3-12b-pt-text"`

The default is all three.

## Prerequisites

- The repo is cloned to `~/multilingual_icm/ICM_work_` (branch `fir-labelling`). `setup_runs.sh` builds both environments if they're missing.
- A Hugging Face login on Fir, with an account that has accepted the licences of `mistralai/Mistral-Nemo-Base-2407` and `google/gemma-3-12b-pt`:
  ```
  HF_HOME=/scratch/$USER/cache/hf ~/venvs/multilingual-icm/bin/hf auth login
  ```

## Steps

1. `bash fir_runs/setup_runs.sh` (login node). It builds the training environment (torch 2.13.0, transformers 4.57.6, peft 0.19.1, accelerate 1.14.0, from PyPI via uv), unpacks and verifies the 7 training/test files, and downloads Mistral and Gemma (~42 GB).
2. `bash fir_runs/submit_runs.sh` for the dry run. Fit `TRAIN_CPUS`/`TRAIN_MEM`/`EVAL_CPUS`/`EVAL_MEM` to Fir's nodes if it complains; these never change results.
3. `SUBMIT=1 bash fir_runs/submit_runs.sh` submits:

| Step | Jobs | Resources |
|---|---|---|
| prepare: verify Mistral + Gemma against Rorqual's checksums; extract the text-only Gemma and verify it | 1 | CPU |
| Mistral: gold_filtered s2; method s1, s2; method_lenmatch s1 | 4 | 2 GPUs, ~5 h each |
| Qwen: method s1, s2; method_lenmatch s1 | 3 | 2 GPUs |
| Gemma: own_unlabelled s1, s2; gold_filtered s1, s2; method s1, s2; method_lenmatch s1 | 7 | 2 GPUs |
| Evaluations: Mistral (4 adapters), Qwen wave 3, Gemma wave 2, Gemma wave 3 | 4 | 1 GPU each |

Every evaluation also scores the base model. `python fir_runs/check_stack.py` compares that against Rorqual's base
scores (`reference/`); identical stacks give identical numbers.

## Send back

- `~/multilingual_icm/data/findings/results/core24_*_{fir,wave2,wave3}.json`
- the `core24-train-*` and `core24-eval-*` logs

The adapters stay on Fir.

## Not here

- The Qwen wave-2 and Mistral wave-2 (own_unlabelled s1/s2, gold_filtered s1) evaluations run on Rorqual, where those adapters are.
