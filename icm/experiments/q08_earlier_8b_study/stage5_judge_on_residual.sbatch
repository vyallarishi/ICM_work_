#!/usr/bin/env bash
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=stage5-judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=124G
#SBATCH --time=3:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out

set -Eeuo pipefail

module reset
module load python/3.11
module load cuda

: "${SCRATCH:?SCRATCH is not set}"
PROJECT_STORAGE="${PROJECT:-${HOME}}"
VENV_DIR="${VENV_DIR:-${PROJECT_STORAGE}/venvs/sm-icm}"

export HF_HOME="${HF_HOME:-${SCRATCH}/sm-icm/hf-cache}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "${VENV_DIR}/bin/activate"
cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"

# Fail fast on any logic regression before touching the GPU.
python -m icm.experiments.q08_earlier_8b_study.stage5_judge_on_residual --self-test

python -m icm.experiments.q08_earlier_8b_study.stage5_judge_on_residual
