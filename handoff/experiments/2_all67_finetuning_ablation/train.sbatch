#!/usr/bin/env bash
# Experiment 2 - LoRA fine-tuning of Qwen3-14B-Base on one training arm, all 67 languages.
# Recipe (fixed, identical to every other run in the paper): LoRA r=16, alpha=32 on all linear layers,
# lr 2e-4, 1 epoch, max length 1,024 tokens (longer examples dropped and counted in the log),
# seed 20261002 + SEED, 4 GPUs (torchrun). Submitted by the run scripts.
#SBATCH --job-name=all67-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
set -Eeuo pipefail
: "${ARM:?}"; : "${SEED:?}"
source "$BUNDLE/setup/job_env.sh" train
DATA="$BUNDLE/data/all67_ablation/sft_$ARM.jsonl"
[ -f "$DATA" ] || { echo "missing training file $DATA" >&2; exit 1; }
OUT="$OUTPUT_DIR/all67_ablation/adapters/${ARM}_s${SEED}${SMOKE:+_smoke}"
mkdir -p "$OUT"
echo "[all67-train] arm $ARM seed $SEED data $(sha256sum "$DATA" | cut -c1-16) -> $OUT ${SMOKE:+(SMOKE: 20 steps)}"
MODEL=$(stage_model Qwen3-14B-Base)
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
torchrun --standalone --nproc_per_node "$NGPU" -m icm.experiments.q09_does_the_dataset_beat_translation.train_lora \
    --model-dir "$MODEL" --data "$DATA" --out "$OUT" --seed $((20261002 + SEED)) ${SMOKE:+--max-steps 20}
[ -f "$OUT/adapter_config.json" ] || { echo "no adapter written" >&2; exit 1; }
echo "[all67-train] done: $OUT"; date
