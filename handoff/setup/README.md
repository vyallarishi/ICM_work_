# Setup

## 1. Settings: `config.env`

Edit these settings once:

| Setting | What to put |
|---|---|
| `ACCOUNT_GPU`, `ACCOUNT_CPU` | your Slurm accounts |
| `MODELS_DIR`, `OUTPUT_DIR`, `CACHE_DIR` | a large filesystem, ~100 GB free (e.g. /scratch). Keep `CACHE_DIR` off /home: vLLM writes many small files. |
| `VENV_VLLM`, `VENV_TRAIN` | where the two Python environments live |
| `MODULES_VLLM`, `MODULES_TRAIN` | environment modules to load in jobs; leave empty if your cluster has no `module` command |
| `GPU_TYPE` | the GPU name your Slurm uses (e.g. `h100`, `a100`); jobs assume 80 GB of GPU memory |

## 2. Two Python environments (Python 3.11)

Two environments are needed because vLLM and the training stack pin different torch and transformers versions.

```
python3.11 -m venv $VENV_VLLM  && $VENV_VLLM/bin/pip install -r setup/requirements-vllm.txt
python3.11 -m venv $VENV_TRAIN && $VENV_TRAIN/bin/pip install -r setup/requirements-train.txt
```

The `requirements-*.txt` files pin the packages that affect results. `full_freeze_*_env.txt` lists every package of the original environments, for debugging only.

## 3. Models

Run:

```
bash setup/download_models.sh
```

It downloads Qwen3-14B-Base and Gemma 3 12B into `$MODELS_DIR`, then builds `gemma-3-12b-pt-text`, the text-only Gemma that experiment 1 uses. For Gemma, first accept the licence at https://huggingface.co/google/gemma-3-12b-pt and run `hf auth login`.

## 4. Check

Run:

```
bash setup/check_setup.sh
```

It checks:
- the settings;
- the data checksums;
- that both models are present;
- the pinned package versions in both environments;
- that Slurm accepts your accounts.

## Files

| File | What it is |
|---|---|
| `job_env.sh` | sourced by every job: loads modules, activates the right environment, points caches at `$CACHE_DIR`, and copies models to node-local disk before loading (reading weights straight from a parallel filesystem can stall for hours) |
| `extract_text_model` | lives in `code/icm/experiments/q09_does_the_dataset_beat_translation/extract_text_model.py`; called by `download_models.sh` |
