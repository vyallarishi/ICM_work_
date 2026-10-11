#!/usr/bin/env bash
# Experiment 2, step 2 (LATER): the "method" arm, one of the model's own chains agreeing with our
# label-free label. Needs data/all67_ablation/sft_method.jsonl, which is produced on the original
# cluster after the 67-language labelling and sent to you. Same recipe, seed and evaluation as step 1.
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/config.env"
[ -f "$BUNDLE/data/all67_ablation/sft_method.jsonl" ] || { echo "sft_method.jsonl not here yet (it comes after labelling)"; exit 1; }
cd "$BUNDLE/experiments/2_all67_finetuning_ablation"; mkdir -p logs
T=$(sbatch --parsable --account="$ACCOUNT_GPU" --gpus-per-node="$GPU_TYPE:4" --cpus-per-task=32 --mem=256G --time=16:00:00 \
    --job-name=all67-train-method --output="logs/%x-%j.out" --export=ALL,ARM=method,SEED=1 train.sbatch)
echo "train method s1  $T"
echo "eval  method s1  $(sbatch --parsable --account="$ACCOUNT_GPU" --gpus-per-node="$GPU_TYPE:1" --cpus-per-task=16 --mem=124G --time=8:00:00 \
    --job-name=all67-eval-method --output="logs/%x-%j.out" --dependency=afterok:$T \
    --export=ALL,ADAPTERS=method_s1,TAG=method_s1 evaluate.sbatch)"
