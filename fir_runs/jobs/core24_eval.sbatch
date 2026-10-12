#!/usr/bin/env bash
# Core-24 evaluation (GPU 1 x H100): base + the given adapters on the pipeline test set
# (1,319 x 25), MGSM human translations and the 42 held-out extension languages (1,319 x 42); q09 evaluate.py unchanged. -> rrg-zhijing
#     STUDENT=Qwen3-14B-Base ADAPTERS="translated_s1 translated_s2" TAG=gold sbatch core24_eval.sbatch
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=core24-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=124G
#SBATCH --time=24:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/findings/logs/%x-%j.out
set +e
module reset >/dev/null 2>&1
module load python/3.11 cuda
set -Eeuo pipefail
: "${STUDENT:?}"; : "${ADAPTERS:?}"; : "${TAG:?}"
PROJECT_STORAGE="${PROJECT:-${HOME}}"
J=/home/rvyalla/multilingual_icm/data/findings/jobs
D=/home/rvyalla/multilingual_icm/data/findings/runs_core24
R=/home/rvyalla/multilingual_icm/data/findings/results
A=/scratch/rvyalla/multilingual-icm/results/${LAYER:-core24}/adapters_${STUDENT}
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=/scratch/rvyalla/cache/vllm TRITON_CACHE_DIR=/scratch/rvyalla/cache/triton XDG_CACHE_HOME=/scratch/rvyalla/cache/xdg
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
# stage the model on node-local disk: lazy mmap reads straight from /scratch (Lustre) can crawl
# at < 1 MB/s (three jobs hung 1.5 h this way on 2026-10-09)
: "${SLURM_TMPDIR:?}"
echo "staging $STUDENT to $SLURM_TMPDIR"; date
rsync -a "/scratch/rvyalla/multilingual-icm/assets/models/$STUDENT/" "$SLURM_TMPDIR/model/"
date
ARGS=""
for name in $ADAPTERS; do
    [ -f "$A/$name/adapter_config.json" ] || { echo "missing adapter $A/$name" >&2; exit 1; }
    ARGS="$ARGS --adapter $name=$A/$name"
done
echo "pre-registration on record: $(sha256sum "$J/PREREG_4_languages.md")"; date
for SET in ${SETS:-pipeline mgsm_human heldout}; do
    python -u -m icm.experiments.q09_does_the_dataset_beat_translation.evaluate $ARGS \
        --model-dir "$SLURM_TMPDIR/model" \
        --eval "$D/eval_$SET.jsonl" --out "$R/core24_${STUDENT}_${SET}_${TAG}.json"
done
date; echo DONE
