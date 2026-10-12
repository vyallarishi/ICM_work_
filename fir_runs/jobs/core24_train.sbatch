#!/usr/bin/env bash
# Core-24 LoRA fine-tuning (GPU 4 x H100), the q09 recipe unchanged (r16/a32, lr 2e-4, 1 epoch,
# max_len 1024); each arm length-filtered on its own tokens. -> rrg-zhijing
#     STUDENT=Qwen3-14B-Base ARM=translated SEED=1 [LAYER=all67] sbatch core24_train.sbatch
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=core24-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/findings/logs/%x-%j.out
set +e
module reset >/dev/null 2>&1
module load python/3.11 cuda
set -Eeuo pipefail
: "${STUDENT:?set STUDENT=Qwen3-14B-Base|Mistral-Nemo-Base-2407|gemma-3-12b-pt-text}"
: "${ARM:?set ARM=translated (later: gold_filtered, own_unlabelled, method)}"
: "${SEED:?set SEED=1|2}"
PROJECT_STORAGE="${PROJECT:-${HOME}}"
J=/home/rvyalla/multilingual_icm/data/findings/jobs
D=/home/rvyalla/multilingual_icm/data/findings/runs_${LAYER:-core24}
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/scratch/rvyalla/cache/triton XDG_CACHE_HOME=/scratch/rvyalla/cache/xdg
source "${PROJECT_STORAGE}/venvs/sm-icm/bin/activate"
OUT=/scratch/rvyalla/multilingual-icm/results/${LAYER:-core24}/adapters_${STUDENT}/${ARM}_s${SEED}${SMOKE:+_smoke}
mkdir -p "$OUT"
# stage the model on node-local disk: lazy mmap reads straight from /scratch (Lustre) can crawl
# at < 1 MB/s (three jobs hung 1.5 h this way on 2026-10-09)
: "${SLURM_TMPDIR:?}"
echo "staging $STUDENT to $SLURM_TMPDIR"; date
rsync -a "/scratch/rvyalla/multilingual-icm/assets/models/$STUDENT/" "$SLURM_TMPDIR/model/"
date
echo "pre-registration on record: $(sha256sum "$J/PREREG_4_languages.md")"; echo "data: $(sha256sum "$D/sft_$ARM.jsonl")"; date
# NGPU (default 4) GPUs; gradient accumulation scales so every update still sees 32 examples
# (per-device batch 4 x accumulation x GPUs = 32), the q09 recipe on any GPU count.
NGPU=${NGPU:-4}; [ $((32 % (4 * NGPU))) -eq 0 ] || { echo "NGPU=$NGPU does not divide 32/4" >&2; exit 1; }
echo "[train] $NGPU GPU(s), grad-accum $((8 / NGPU)), effective batch 32"
torchrun --standalone --nproc_per_node $NGPU -m icm.experiments.q09_does_the_dataset_beat_translation.train_lora \
    --model-dir "$SLURM_TMPDIR/model" --data "$D/sft_$ARM.jsonl" --out "$OUT" \
    --seed $((20261002 + SEED)) --grad-accum $((8 / NGPU)) ${SMOKE:+--max-steps 20}
date; echo DONE
