# Files: what each one is and where it came from

"Original" paths refer to the project on the original cluster (`~/multilingual_icm`).

## Top level
| Path | What it is |
|---|---|
| `README.md` | start here: steps, what to send back, ground rules |
| `config.env` | the only file to edit: accounts, paths, environments, GPU type |
| `FILES.md` | this file |

## setup/
| Path | What it is |
|---|---|
| `README.md` | how to fill config.env, build the two environments and get the models |
| `requirements-vllm.txt`, `requirements-train.txt` | pinned package versions of the original runs |
| `full_freeze_vllm_env.txt`, `full_freeze_train_env.txt` | every package of the original environments (reference only) |
| `download_models.sh` | downloads Qwen3-14B-Base and Gemma 3 12B; builds the text-only Gemma |
| `check_setup.sh` | pre-flight check (settings, data, models, package versions, Slurm accounts) |
| `build_data.py`, `build_data.sbatch` | rebuilds `data/` from pinned public sources (GSM8K-106 on Hugging Face, MGSM on GitHub) and verifies it |
| `fingerprints/` | what `build_data.py` must reproduce: a sha256 per dataset table, per derived file, and MGSM's MD5s |
| `job_env.sh` | sourced by every job (modules, environment, caches, node-local model copy) |

## experiments/1_gemma_generation/
See its README. New files written for this handoff; they call the unchanged generation code in `code/generation`.

## experiments/2_all67_finetuning_ablation/
See its README. New files written for this handoff; they call the unchanged `train_lora.py` / `evaluate.py` in `code/icm`.

## code/
| Path | Original | Changes |
|---|---|---|
| `code/generation/` | `ICM_work_/generation/` (scripts, prompts, configs, tests, docs) | none; its old `cluster/` job scripts are left out (replaced by experiments/1) |
| `code/icm/paths.py` | `ICM_work_/icm/paths.py` | default output folder now inside this bundle (jobs set `ICM_RUN_DIR` anyway) |
| `code/icm/experiments/q09_does_the_dataset_beat_translation/train_lora.py` | same path in `ICM_work_` | only the default `MODEL_DIR` string, which pointed at the original cluster (jobs always pass `--model-dir`) |
| `code/icm/experiments/q09_does_the_dataset_beat_translation/evaluate.py` | same | same one-line change |
| `code/icm/experiments/q09_does_the_dataset_beat_translation/extract_text_model.py` | same | none |
| `code/icm/**/__init__.py` | same | none (empty) |

## data/ (built by `setup/build_data.py`; only the two small JSON configs and language_selection.json are in git)
| Path | Original | What it is |
|---|---|---|
| `gsm8k-106/` | `/scratch/.../assets/datasets/gsm8k-106` (open-multilang/gsm8k-106) | GSM8K translated into 106 languages, train + test splits |
| `gemma_generation/language_chunks.json` | `data/findings/jobs/gen_chunks.json` (gemma chunks) | the 25 languages split into 4 jobs |
| `gemma_generation/generation_settings.json` | `data/findings/jobs/gen_manifest_longctx.json` | sampling regimes (= the Qwen record's `best_regimes_v2`, context 6,144) |
| `all67_ablation/sft_translated.jsonl` | `data/findings/runs_all67/sft_translated.jsonl` | training set of the translated arm (500,691 examples) |
| `all67_ablation/sft_method.jsonl` | (arrives later) | training set of the method arm |
| `eval_sets/test_core25_languages.jsonl` | `data/findings/runs_core24/eval_pipeline.jsonl` | 1,317 test problems × 25 languages |
| `eval_sets/test_42_extension_languages.jsonl` | `data/findings/runs_core24/eval_heldout.jsonl` | 1,317 × 42 languages (one broken Khmer item removed) |
| `eval_sets/test_mgsm_human_translations.jsonl` | `data/findings/runs_core24/eval_mgsm_human.jsonl` | MGSM human translations, 248 × 11 languages |
| `language_selection.json` | `data/findings/results/language_selection.json` | the pre-registered language sets and tiers (reference) |

## results/
Experiment 2 writes its accuracy files to `results/all67_ablation/`. Experiment 1's output is large and goes to `$OUTPUT_DIR/gemma_generation/final/`.
