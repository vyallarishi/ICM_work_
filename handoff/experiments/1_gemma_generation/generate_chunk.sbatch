#!/usr/bin/env bash
# Experiment 1 - generate Gemma 3 12B chains for ONE chunk of languages.
# One vLLM copy per GPU, languages split across the GPUs; 20 chains per problem (8 at T0.2, 8 at T0.5,
# 4 at T0.8), native 2-shot prompt, 768 new tokens: the same pipeline and settings as the Qwen3 14B
# record, plus end-of-text suppressed for the first $MIN_TOKENS tokens. Submitted by the run scripts.
#SBATCH --job-name=gemma-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
set -Eeuo pipefail
: "${CHUNK:?}"; : "${LANGUAGES:?}"; : "${MIN_TOKENS:?}"
export MIN_TOKENS
source "$BUNDLE/setup/job_env.sh" vllm
GEN="$BUNDLE/code/generation"
RESULT_ROOT="$OUTPUT_DIR/gemma_generation/chunk_$CHUNK"
MANIFEST="$BUNDLE/data/gemma_generation/generation_settings.json"
mkdir -p "$RESULT_ROOT/logs"
echo "[gemma-gen] chunk $CHUNK: $LANGUAGES | limit ${LIMIT_PER_LANGUAGE:-all} | min_tokens $MIN_TOKENS"
MODEL=$(stage_model gemma-3-12b-pt-text)
DATA="$LOCAL_TMP/gsm8k-106"; rsync -a "$BUNDLE/data/gsm8k-106/" "$DATA/"
cd "$GEN"
python scripts/preflight_offline.py --model-dir "$MODEL" --dataset-dir "$DATA" --language en
IFS=',' read -r -a GPU_IDS <<<"${CUDA_VISIBLE_DEVICES:?no GPUs allocated}"
GPU_COUNT="${#GPU_IDS[@]}"
SHARDS="$RESULT_ROOT/workers"
pids=()
for i in "${!GPU_IDS[@]}"; do
    cmd=(python "$BUNDLE/experiments/1_gemma_generation/min_tokens/run_experiment_mintok.py"
         --manifest "$MANIFEST" --model "$MODEL" --dataset "$DATA" --output-dir "$SHARDS/shard_$i"
         --language-shard-count "$GPU_COUNT" --language-shard-index "$i" --generate-only --languages $LANGUAGES)
    [ -n "${LIMIT_PER_LANGUAGE:-}" ] && cmd+=(--limit-per-language "$LIMIT_PER_LANGUAGE")
    mkdir -p "$SHARDS/shard_$i"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" "${cmd[@]}" > "$RESULT_ROOT/logs/shard-$i.log" 2>&1 &
    pids+=("$!")
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
[ "$status" -eq 0 ] || { echo "a GPU worker failed; see $RESULT_ROOT/logs/" >&2; exit 1; }
for f in "$RESULT_ROOT"/logs/shard-*.log; do
    grep -q "min_tokens=$MIN_TOKENS" "$f" || { echo "min_tokens marker missing in $f: the rule was NOT applied" >&2; exit 1; }
done
python "$BUNDLE/experiments/1_gemma_generation/check_chains.py" "$SHARDS"
echo "[gemma-gen] chunk $CHUNK done: $RESULT_ROOT"; date
