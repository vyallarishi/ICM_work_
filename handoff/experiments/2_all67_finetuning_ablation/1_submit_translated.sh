#!/usr/bin/env bash
# Experiment 2, step 1: the "translated" arm (translated gold solutions, 500,691 examples over 67
# languages), seed 1: training (4 GPUs, ~7-8 h) then evaluation (1 GPU, ~2-3 h).
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/config.env"
cd "$BUNDLE/experiments/2_all67_finetuning_ablation"; mkdir -p logs
T=$(sbatch --parsable --account="$ACCOUNT_GPU" --gpus-per-node="$GPU_TYPE:4" --cpus-per-task=32 --mem=256G --time=16:00:00 \
    --job-name=all67-train-translated --output="logs/%x-%j.out" --export=ALL,ARM=translated,SEED=1 train.sbatch)
echo "train translated s1  $T"
echo "eval  translated s1  $(sbatch --parsable --account="$ACCOUNT_GPU" --gpus-per-node="$GPU_TYPE:1" --cpus-per-task=16 --mem=124G --time=8:00:00 \
    --job-name=all67-eval-translated --output="logs/%x-%j.out" --dependency=afterok:$T \
    --export=ALL,ADAPTERS=translated_s1,TAG=translated_s1 evaluate.sbatch)"
