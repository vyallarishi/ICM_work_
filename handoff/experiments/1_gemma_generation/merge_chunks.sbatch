#!/usr/bin/env bash
# Experiment 1 - merge the 4 chunks into ONE record ($OUTPUT_DIR/gemma_generation/final) with the
# generation repo's own merge (pooling, statistics, candidate views), unchanged. CPU only.
#SBATCH --job-name=gemma-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=%x-%j.out
set -Eeuo pipefail
source "$BUNDLE/setup/job_env.sh" vllm
G="$OUTPUT_DIR/gemma_generation"
S="$G/all_shards"; rm -rf "$S"; mkdir -p "$S"
for C in "$G"/chunk_[0-9]*; do   # numbered chunks only (not chunk_smoke)
    for sh in "$C"/workers/shard_*; do ln -s "$sh" "$S/shard_$(basename "$C")_${sh##*shard_}"; done   # the merge looks for shard_* dirs
done
ls "$S"
cd "$BUNDLE/code/generation"
python -u scripts/merge_experiment_shards.py --shards-root "$S" --output-dir "$G/final" --overwrite
python "$BUNDLE/experiments/1_gemma_generation/check_chains.py" "$G/final/pooled"
echo "[gemma-merge] record: $G/final"; date
