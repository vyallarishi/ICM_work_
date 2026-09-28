#!/usr/bin/env bash
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=stage22-regen14b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=124G
#SBATCH --time=2:00:00
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

source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"

python -m icm.experiments.q03_does_regeneration_help.regeneration_15lang --self-test
python -m icm.experiments.q03_does_regeneration_help.regeneration_15lang --phase gen --dry-run
python -m icm.experiments.q03_does_regeneration_help.regeneration_15lang --phase gen
python -m icm.experiments.q03_does_regeneration_help.regeneration_15lang --phase decide
