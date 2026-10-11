#!/usr/bin/env bash
# Experiment 2 - accuracy of the base model and the given adapters (greedy decoding, prompt
# "Question: ... Answer:", 1,024 new tokens, exact match on the final number) on the three test sets:
#   test_core25_languages         full GSM8K test split x 24 core languages + English
#   test_42_extension_languages   full GSM8K test split x the other 42 languages
#   test_mgsm_human_translations  MGSM human translations (10 languages + English)
# Writes $BUNDLE/results/all67_ablation/<test set>__<TAG>.json. Submitted by the run scripts.
#SBATCH --job-name=all67-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
set -Eeuo pipefail
: "${ADAPTERS:?}"; : "${TAG:?}"
source "$BUNDLE/setup/job_env.sh" vllm
A="$OUTPUT_DIR/all67_ablation/adapters"
ARGS=""
for name in $ADAPTERS; do
    [ -f "$A/$name/adapter_config.json" ] || { echo "missing adapter $A/$name" >&2; exit 1; }
    ARGS="$ARGS --adapter $name=$A/$name"
done
MODEL=$(stage_model Qwen3-14B-Base)
R="$BUNDLE/results/all67_ablation"; mkdir -p "$R"
for SET in ${SETS:-test_core25_languages test_42_extension_languages test_mgsm_human_translations}; do
    python -u -m icm.experiments.q09_does_the_dataset_beat_translation.evaluate $ARGS --model-dir "$MODEL" \
        --eval "$BUNDLE/data/eval_sets/$SET.jsonl" --out "$R/${SET}__${TAG}.json" ${LIMIT_PER_LANG:+--limit-per-lang $LIMIT_PER_LANG}
done
echo "[all67-eval] results in $R"; date
