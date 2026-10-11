#!/usr/bin/env bash
# Experiment 1, step 0 (~30-60 min on 1 GPU): 9 languages x 30 problems through the FULL generation
# path (same job script as the real run). Passes if the job ends with "OK" from check_chains.py:
# no language above 5% empty chains. For reference, on our cluster this setup gave 0% empty chains,
# ~87-92% with an answer line and ~40% accuracy (Gemma 3 12B base).
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/config.env"
cd "$BUNDLE/experiments/1_gemma_generation"; mkdir -p logs
sbatch --account="$ACCOUNT_GPU" --gpus-per-node="$GPU_TYPE:1" --cpus-per-task=16 --mem=124G --time=2:00:00 \
    --job-name=gemma-gen-smoke --output="logs/%x-%j.out" \
    --export=ALL,CHUNK=smoke,MIN_TOKENS=8,LIMIT_PER_LANGUAGE=30,LANGUAGES="en de fr ja sw te kn hy uz" generate_chunk.sbatch
echo "watch: experiments/1_gemma_generation/logs/gemma-gen-smoke-<jobid>.out (expect a final 'OK')"
