# Multilingual ICM Candidate Generation

This repository samples multilingual GSM8K reasoning chains from Qwen3-8B-Base. Generation,
gold-based analysis, and ICM dataset construction are separate stages so the primary ICM inputs can
remain target-label-blind.

For Compute Canada Rorqual, start with [docs/rorqual.md](docs/rorqual.md).

## Output Contract

An experiment has four intentionally distinct artifact groups:

```text
results/<experiment>/
  runs/<regime>/generated/<language>.jsonl
  pooled/generated/<language>.jsonl
  analysis/item_candidate_stats/<language>.jsonl
  analysis/candidate_stats_by_language.csv
  analysis/candidate_stats_by_regime.csv
  analysis/candidate_stats_by_source.csv
  candidate_views/all_usable/<language>.jsonl
  candidate_views/random_4/<language>.jsonl
  candidate_views/diverse_4/<language>.jsonl
  candidate_views/oracle_balanced_4/<language>.jsonl  # only when explicitly requested
```

- `runs/` contains immutable per-regime generations.
- `pooled/` contains deduplicated candidates with parsed answers and gold-derived `correct` values.
  Use it for auditing, never as a label-blind ICM input.
- `analysis/` contains per-item and per-language correct/incorrect statistics.
- `candidate_views/` contains derived ICM datasets. Gold answers and candidate `correct` fields are
  removed by default.

`all_usable` retains every parseable, nondegenerate, deduplicated candidate. With the main manifest,
up to 20 natural candidates are generated per item before quality filtering and cross-regime
deduplication. `random_4` is a deterministic label-blind sample. `diverse_4` covers distinct
model-predicted answers first and then generation regimes/text diversity; it never compares an
answer with gold. `oracle_balanced_4` uses gold correctness and is a supervised control.

## Setup

```bash
uv sync
```

## Few-Shot Main Run

The current primary manifest uses fully native six-shot prompts and 20 natural samples per item:

- 8 samples at temperature 0.2.
- 8 samples at temperature 0.7.
- 4 samples at temperature 1.0.

Each language uses demonstrations from test-split `original_id` values
`(0, 5, 7, 10, 13, 20)`. Training targets are from the train split.

Pilot:

```bash
uv run python scripts/run_experiment.py \
  --manifest configs/pilot_regimes.json \
  --output-dir results/pilot_qwen3_8b_icm \
  --languages en hi sw ar fr ja bn te af \
  --limit-per-language 200
```

Full run:

```bash
uv run python scripts/run_experiment.py \
  --manifest configs/main_regimes.json \
  --output-dir results/qwen3_8b_icm_train
```

Inspect correctness without feeding it to ICM:

```bash
column -s, -t < results/qwen3_8b_icm_train/analysis/candidate_stats_by_language.csv | less -S
```

## Rebuild Dataset Views

The derived views can be rebuilt without rerunning the model:

```bash
uv run python scripts/build_candidate_views.py \
  --input-dir results/qwen3_8b_icm_train/pooled \
  --output-dir results/qwen3_8b_icm_train \
  --views all_usable random_4 diverse_4 \
  --overwrite
```

Build the supervised oracle control explicitly:

```bash
uv run python scripts/build_candidate_views.py \
  --input-dir results/qwen3_8b_icm_train/pooled \
  --output-dir results/qwen3_8b_icm_train \
  --views all_usable random_4 diverse_4 oracle_balanced_4 \
  --overwrite
```

The default view files are sanitized. `--include-gold-in-views` exists only for audits and should
not be used for ICM convergence inputs.

## Zero-Shot Ablation

Zero-shot prompting is already supported by the same runner. It requires only
`prompt_template: "zero_shot_cot"` and `few_shot_count: 0`; a ready manifest is included:

```bash
uv run python scripts/run_experiment.py \
  --manifest configs/zero_shot_regimes.json \
  --output-dir results/qwen3_8b_icm_zero_shot
```

The main and pilot manifests remain few-shot as requested.

## Label-Blind Adaptive Retry

```bash
uv run python scripts/run_adaptive_retry.py \
  --base-dir results/qwen3_8b_icm_train \
  --manifest configs/retry_regimes.json
```

Default retries use only observable candidate properties:

- Fewer than four usable candidates triggers low-temperature native few-shot sampling.
- Fewer than two distinct predicted answers triggers high-temperature native few-shot sampling.

Retries never trigger on whether a candidate is correct. Retry outputs keep
`candidate_source = "retry"`, and the original main generations remain unchanged.

## Wrong-Chain Ablation

Deliberately wrong prompting is isolated from the natural-sampling experiment:

```bash
uv run python scripts/run_experiment.py \
  --manifest configs/wrong_chain_ablation.json \
  --output-dir results/qwen3_8b_icm_wrong_chain_ablation
```

## Local Wrapper

```bash
bash jobrun.sh --smoke
bash jobrun.sh --generate
bash jobrun.sh --views
bash jobrun.sh --retry
```

Set `MANIFEST`, `RETRY_MANIFEST`, `OUTPUT_DIR`, `LANGUAGES`, and `LIMIT_PER_LANGUAGE` to customize
the wrapper.
