#!/usr/bin/env bash
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=stage21-ensloop14b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=124G
#SBATCH --time=8:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out

set -Eeuo pipefail

module reset
module load python/3.11
module load cuda

PROJECT_STORAGE="${PROJECT:-${HOME}}"

export HF_HOME="${HF_HOME:-${SCRATCH}/hf-cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"

# phase 1: continuation scores (vLLM venv) -- once, lap-independent
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
python -m icm.experiments.q02_can_the_model_judge_itself.ensemble_judge_in_loop --self-test
python -m icm.experiments.q02_can_the_model_judge_itself.ensemble_judge_in_loop --phase cc --dry-run
python -m icm.experiments.q02_can_the_model_judge_itself.ensemble_judge_in_loop --phase cc
deactivate

# phase 2: the 4-arm loop (transformers venv), reads the cc cache
source "${PROJECT_STORAGE}/venvs/sm-icm/bin/activate"
python -m icm.experiments.q02_can_the_model_judge_itself.ensemble_judge_in_loop --phase loop --dry-run
python -m icm.experiments.q02_can_the_model_judge_itself.ensemble_judge_in_loop --phase loop
