#!/usr/bin/env bash
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=stage41-markers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=124G
#SBATCH --time=4:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
set -Eeuo pipefail
module reset
module load python/3.11
module load cuda
PROJECT_STORAGE="${PROJECT:-${HOME}}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${PROJECT_STORAGE}/venvs/sm-icm/bin/activate"
cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
python -m icm.experiments.q02_can_the_model_judge_itself.marker_token_comparison --self-test
python -m icm.experiments.q02_can_the_model_judge_itself.marker_token_comparison --dry-run
python -m icm.experiments.q02_can_the_model_judge_itself.marker_token_comparison
