#!/usr/bin/env bash
# Q09 step 1 (CPU): build the two matched training sets and the eval set. -> def-zhijing
#SBATCH --account=def-zhijing
#SBATCH --job-name=q09-build-sets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
module reset
module load python/3.11
set -Eeuo pipefail
PROJECT_STORAGE="${PROJECT:-${HOME}}"
cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
export HF_DATASETS_OFFLINE=1
python -u -m icm.experiments.q09_does_the_dataset_beat_translation.build_sft_sets
