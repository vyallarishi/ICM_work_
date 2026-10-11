#!/usr/bin/env bash
# Experiment 2, step 0 (~1 h): 20 training steps on the translated arm, then the evaluation of that
# adapter on 3 problems per language of every test set. Passes if both jobs finish and
# results/all67_ablation/*__smoke.json exist. Smoke outputs are throwaway (adapter name ends in _smoke).
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/config.env"
cd "$BUNDLE/experiments/2_all67_finetuning_ablation"; mkdir -p logs
T=$(sbatch --parsable --account="$ACCOUNT_GPU" --gpus-per-node="$GPU_TYPE:4" --cpus-per-task=32 --mem=256G --time=1:00:00 \
    --job-name=all67-train-smoke --output="logs/%x-%j.out" --export=ALL,ARM=translated,SEED=1,SMOKE=1 train.sbatch)
echo "train smoke $T"
echo "eval  smoke $(sbatch --parsable --account="$ACCOUNT_GPU" --gpus-per-node="$GPU_TYPE:1" --cpus-per-task=16 --mem=124G --time=1:00:00 \
    --job-name=all67-eval-smoke --output="logs/%x-%j.out" --dependency=afterok:$T \
    --export=ALL,ADAPTERS=translated_s1_smoke,TAG=smoke,LIMIT_PER_LANG=3 evaluate.sbatch)"
