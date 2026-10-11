# Running on Rorqual

The recommended layout for `Qwen3-8B-Base` is one independent vLLM process per H100. The BF16 model
fits easily on one 80 GB H100, so four replicas over disjoint language shards provide better
throughput than four-way tensor parallelism. The supplied job adapts automatically when allocated
one, two, or four GPUs.

## 1. Clone and configure storage

Run these commands on a login node. Replace the project path if your allocation uses a different
directory.

```bash
export PROJECT=$HOME/projects/rrg-zhijing/$USER
export REPO_ROOT=$PROJECT/multilingual_icm
export VENV_DIR=$PROJECT/venvs/multilingual-icm
export ASSET_ROOT=$SCRATCH/multilingual-icm/assets

git clone <YOUR_REPOSITORY_URL> "$REPO_ROOT"
cd "$REPO_ROOT"
```

If the model requires authentication, export `HF_TOKEN` before setup. Do not store the token in the
repository or Slurm script.

## 2. Build the environment and download assets

```bash
cd "$REPO_ROOT"
bash cluster/setup_rorqual.sh
```

The setup script:

- Loads Python 3.11 and CUDA modules.
- Installs `uv` under `$HOME/.local/bin` when needed.
- Creates the environment at `$VENV_DIR`.
- Stores the Hugging Face and uv caches under `$SCRATCH`.
- Downloads `Qwen/Qwen3-8B-Base` into a self-contained model directory.
- Downloads both `train` and `test` for all 106 languages and saves them with `Dataset.save_to_disk`.

Compute jobs use `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`; they do
not contact Hugging Face.

## 3. Run a short pilot

The `logs` directory must exist before `sbatch`; the setup script creates it.

```bash
cd "$REPO_ROOT"
sbatch \
  --gpus-per-node=h100:2 \
  --cpus-per-task=32 \
  --mem=128G \
  --time=02:00:00 \
  --export=ALL,EXPERIMENT_NAME=rorqual-pilot,MANIFEST=$REPO_ROOT/configs/pilot_regimes.json,LANGUAGES="af en",LIMIT_PER_LANGUAGE=20 \
  cluster/rorqual_generate.sbatch
```

Monitor the allocation and per-GPU workers:

```bash
squeue -u "$USER"
tail -f logs/icm-qwen3-8b-<JOB_ID>.out
tail -f $SCRATCH/multilingual-icm/results/rorqual-pilot/logs/shard-0-<JOB_ID>.log
```

Pilot results are written to:

```text
$SCRATCH/multilingual-icm/results/rorqual-pilot/final/
```

Confirm `preflight=ok`, inspect a few rendered prompts in raw JSONL records, and check
`final/analysis/candidate_stats_by_language.csv` before the full run.

## 4. Run the full four-H100 experiment

```bash
cd "$REPO_ROOT"
sbatch cluster/rorqual_generate.sbatch
```

The checked-in request is:

```text
1 node, 4 H100s, 64 CPU cores, 256 GB RAM, 24 hours
```

Each GPU gets a deterministic quarter of the languages. Model and dataset reads are staged to
`$SLURM_TMPDIR`; generated language files are checkpointed directly under `$SCRATCH` so completed
work survives a timeout.

The complete experiment is very large. If the job reaches its walltime, submit the same command with
the same `EXPERIMENT_NAME`. Completed language files are skipped and an incomplete language is
regenerated. Keep the GPU count unchanged between resumptions so the deterministic shard assignment
and worker directories remain stable.

To request fewer GPUs, override the Slurm resources. For example:

```bash
sbatch \
  --gpus-per-node=h100:1 \
  --cpus-per-task=16 \
  --mem=64G \
  cluster/rorqual_generate.sbatch
```

## Output layout

```text
$SCRATCH/multilingual-icm/results/<experiment>/
  workers_<N>gpu/shard_<i>/runs/<regime>/generated/<language>.jsonl
  logs/shard-<i>-<job-id>.log
  final/pooled/generated/<language>.jsonl
  final/analysis/item_candidate_stats/<language>.jsonl
  final/analysis/candidate_stats_by_language.csv
  final/candidate_views/all_usable/<language>.jsonl
  final/candidate_views/random_4/<language>.jsonl
  final/candidate_views/diverse_4/<language>.jsonl
```

The main manifest uses six fully localized demonstrations selected from the `test` split by fixed
`original_id` values `(0, 5, 7, 10, 13, 20)`. Training queries come from `train`. Prompt instructions,
demonstration questions, and demonstration reasoning are therefore all in the target language.

## Performance settings

The H100 defaults in `configs/main_regimes.json` are:

```text
tensor_parallel_size=1
gpu_memory_utilization=0.92
max_model_len=4096
max_new_tokens=512
batch_size=64 prompts
max_num_seqs=512
max_num_batched_tokens=32768
enable_prefix_caching=true
```

The outer batch is intentionally 64: for an `n=8` regime it can already create 512 child sequences.
Increasing it does not necessarily improve vLLM throughput and raises host-memory and result-buffer
pressure. Use `nvidia-smi dmon` or Rorqual's monitoring portal during the pilot before tuning these
values upward.

## Adaptive retry job

After the full run has produced `final/analysis`, submit the retry job with the same experiment name:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=qwen3-8b-native-6shot-main cluster/rorqual_retry.sbatch
```

Retry is intentionally configured for one H100, 16 CPU cores, and 64 GB RAM. It only generates for
items with too few usable candidates or too little predicted-answer diversity, follows the original
pooled-source manifest back to the four worker shards, and then rebuilds the pool, analysis, and
candidate views in place. These triggers do not inspect gold correctness.

## Zero-shot ablation

The cluster runner uses whichever manifest is exported. A zero-shot run therefore needs no code
change:

```bash
sbatch \
  --export=ALL,EXPERIMENT_NAME=qwen3-8b-zero-shot,MANIFEST=$REPO_ROOT/configs/zero_shot_regimes.json \
  cluster/rorqual_generate.sbatch
```
