#!/usr/bin/env bash
# Generic evaluation (any Alliance cluster): q09 evaluate.py unchanged; base + the given adapters (name=path
# pairs) on one eval file. Greedy, exact match; per-item correctness is stored in the output.
#     STUDENT=<model dir> EVAL=<eval jsonl> ADAPTERS="a=/path/a b=/path/b" OUT=<result json> sbatch eval.sbatch
#SBATCH --job-name=r3-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set +e
module reset >/dev/null 2>&1
module load python/3.11 cuda
set -Eeuo pipefail
: "${STUDENT:?}"; : "${EVAL:?}"; : "${ADAPTERS:?}"; : "${OUT:?}"
R3=$HOME/multilingual_icm/ICM_work_/fir_round3
export PYTHONPATH="$HOME/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=$SCRATCH/cache/vllm TRITON_CACHE_DIR=$SCRATCH/cache/triton XDG_CACHE_HOME=$SCRATCH/cache/xdg
source "${PROJECT:-$HOME}/venvs/multilingual-icm/bin/activate"
mkdir -p "$HOME/multilingual_icm/data/ICM_work_/runs" "$(dirname "$OUT")"; cd "$HOME/multilingual_icm/data/ICM_work_/runs"
: "${SLURM_TMPDIR:?}"
echo "staging $STUDENT"; date
rsync -a "$SCRATCH/multilingual-icm/assets/models/$STUDENT/" "$SLURM_TMPDIR/model/"; date
ARGS=""
for pair in $ADAPTERS; do path="${pair#*=}"; [ -f "$path/adapter_config.json" ] || { echo "missing adapter $path" >&2; exit 1; }; ARGS="$ARGS --adapter $pair"; done
echo "pre-registration on record: $(sha256sum "$R3/PREREG_round3.md")"; date
python -u -m icm.experiments.q09_does_the_dataset_beat_translation.evaluate $ARGS --model-dir "$SLURM_TMPDIR/model" --eval "$EVAL" --out "$OUT"
date; echo DONE
