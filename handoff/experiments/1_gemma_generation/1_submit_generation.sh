#!/usr/bin/env bash
# Experiment 1, step 1: the real run. 4 chunk jobs (4 GPUs each, ~4-7 h) over the 24 core languages +
# English, full GSM8K train split (7,473 problems), then one CPU merge job into
# $OUTPUT_DIR/gemma_generation/final. Run only after 0_smoke_test.sh passed.
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/config.env"
cd "$BUNDLE/experiments/1_gemma_generation"; mkdir -p logs
CHUNKS="$BUNDLE/data/gemma_generation/language_chunks.json"
IDS=""
for k in 0 1 2 3; do
    LANGS=$(python3 -c "import json; print(' '.join(json.load(open('$CHUNKS'))['chunks'][$k]))")
    id=$(sbatch --parsable --account="$ACCOUNT_GPU" --gpus-per-node="$GPU_TYPE:4" --cpus-per-task=64 --mem=256G \
         --time=12:00:00 --job-name=gemma-gen-c$k --output="logs/%x-%j.out" \
         --export=ALL,CHUNK=$k,MIN_TOKENS=8,LANGUAGES="$LANGS" generate_chunk.sbatch)
    IDS="$IDS:$id"; echo "chunk $k  job $id  ($LANGS)"
done
echo "merge    job $(sbatch --parsable --account="$ACCOUNT_CPU" --dependency=afterok$IDS --output="logs/%x-%j.out" merge_chunks.sbatch)"
