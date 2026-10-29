#!/usr/bin/env bash
# Generic LoRA fine-tuning (any Alliance cluster): the core-24 recipe unchanged (r16/a32, lr 2e-4, 1 epoch,
# max_len 1024, effective batch 32, seed 20261002+SEED), model staged to node-local disk. 2 GPUs by default.
#     STUDENT=<model dir name> DATA=<sft jsonl> OUT=<adapter dir> SEED=1 sbatch [--account=...] train.sbatch
#SBATCH --job-name=r3-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set +e
module reset >/dev/null 2>&1
module load python/3.11 cuda
set -Eeuo pipefail
: "${STUDENT:?}"; : "${DATA:?}"; : "${OUT:?}"; : "${SEED:?}"
R3=$HOME/multilingual_icm/ICM_work_/fir_round3
export PYTHONPATH="$HOME/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$SCRATCH/cache/triton XDG_CACHE_HOME=$SCRATCH/cache/xdg
source "${PROJECT:-$HOME}/venvs/sm-icm/bin/activate"
mkdir -p "$OUT"; : "${SLURM_TMPDIR:?}"
echo "staging $STUDENT"; date
rsync -a "$SCRATCH/multilingual-icm/assets/models/$STUDENT/" "$SLURM_TMPDIR/model/"; date
echo "pre-registration on record: $(sha256sum "$R3/PREREG_round3.md")"; echo "data: $(sha256sum "$DATA")"
NGPU=${NGPU:-2}; [ $((32 % (4 * NGPU))) -eq 0 ] || { echo "NGPU=$NGPU does not divide 32/4" >&2; exit 1; }
echo "[train] $NGPU GPU(s), grad-accum $((8 / NGPU)), effective batch 32"
torchrun --standalone --nproc_per_node $NGPU -m icm.experiments.q09_does_the_dataset_beat_translation.train_lora \
    --model-dir "$SLURM_TMPDIR/model" --data "$DATA" --out "$OUT" --seed $((20261002 + SEED)) --grad-accum $((8 / NGPU)) ${SMOKE:+--max-steps 20}
date; echo DONE
