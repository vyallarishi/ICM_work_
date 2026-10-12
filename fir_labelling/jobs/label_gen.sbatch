#!/usr/bin/env bash
# Labelling, GPU step: vote, then regenerate chains for the contested pools (icm/method, RECORD rules,
# Qwen3-14B-Base, 1 x H100; model staged on node-local disk). -> rrg-zhijing
#     SET=core25 sbatch label_gen.sbatch
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=label-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=186G
#SBATCH --time=24:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/findings/logs/%x-%j.out
set +e
module reset >/dev/null 2>&1
module load python/3.11 cuda
set -Eeuo pipefail
: "${SET:?set SET=core25|all67}"
PROJECT_STORAGE="${PROJECT:-${HOME}}"
J=/home/rvyalla/multilingual_icm/data/findings/jobs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=$SLURM_TMPDIR/vllm TRITON_CACHE_DIR=$SLURM_TMPDIR/triton XDG_CACHE_HOME=$SLURM_TMPDIR/xdg
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
echo "pre-registration on record: $(sha256sum "$J/PREREG_4_languages.md")"; date
rsync -a /scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base/ "$SLURM_TMPDIR/model/"; date
python -u "$J/label_run.py" gen --set "$SET" --model-dir "$SLURM_TMPDIR/model"
date; echo DONE
