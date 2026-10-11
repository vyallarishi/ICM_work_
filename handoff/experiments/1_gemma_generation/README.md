# Experiment 1: Gemma 3 12B generation

## Why

The paper's main generator is Qwen3 14B base. Gemma 3 12B base is the second, from a different model family at a similar size. It lets us test whether errors shared across languages are a property of one model or of the problems. You generate Gemma's chains exactly the way Qwen's were generated.

## What runs

| Setting | Value |
|---|---|
| Languages | the 24 core languages + English (`data/gemma_generation/language_chunks.json`, 4 chunks of 6–7) |
| Problems | all 7,473 GSM8K training problems (`data/gsm8k-106`, the translated dataset) |
| Chains per problem | 20: 8 at temperature 0.2, 8 at 0.5, 4 at 0.8, top-p 0.95 (`data/gemma_generation/generation_settings.json`) |
| Prompt | native-language 2-shot (examples are GSM8K test problems 0 and 5); up to 768 new tokens |
| Generation code | the project's generation repo, unchanged (`code/generation`) |

**The one difference from Qwen: end-of-text is blocked for the first 8 tokens.**
- Without this, Gemma ends about 89% of its answers immediately, producing an empty chain.
- With it, 0% are empty, and the chains are normal (about 40% correct).
- `min_tokens/` applies this rule without touching the generation code.
- Each job fails on purpose if any GPU's log lacks the rule's marker, so the rule can't be silently skipped.

## Steps

| Step | Command | Pass looks like |
|---|---|---|
| 0. Smoke test (1 GPU, under 1 h) | `bash 0_smoke_test.sh` | the log `logs/gemma-gen-smoke-<id>.out` ends with a table per language and `OK`: 0% empty, answer-line ≈ 0.85–0.95, accuracy ≈ 0.3–0.5 |
| 1. Real run | `bash 1_submit_generation.sh` | 4 chunk jobs, each ending with the same table and `OK`; then the merge job prints the table for the merged record and `OK` |

**Result:** `$OUTPUT_DIR/gemma_generation/final/`. Send this whole folder back.

## Files

| File | Role |
|---|---|
| `0_smoke_test.sh`, `1_submit_generation.sh` | submit the jobs (read `config.env`) |
| `generate_chunk.sbatch` | one chunk: copies the model and dataset to node-local disk, runs one generation process per GPU, checks the min-tokens marker, runs `check_chains.py` |
| `merge_chunks.sbatch` | combines the 4 chunks into one record using the generation repo's own merge |
| `check_chains.py` | per-language table: problems, chains per problem, % empty, % with an answer line, accuracy; fails above 5% empty |
| `min_tokens/gen_min_tokens_launch.py` | runs the repo's `generate_chains.py` unchanged, with the min-tokens rule added to every sampling call |
| `min_tokens/run_experiment_mintok.py`, `min_tokens/python_mintok` | route the repo's `run_experiment.py` generation step through the launcher above |
