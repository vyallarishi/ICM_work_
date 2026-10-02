#!/usr/bin/env bash
# H7 step 3 (CPU): label the variants with the pipeline, score against derived answers. -> def-zhijing
#SBATCH --account=def-zhijing
#SBATCH --job-name=h7-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
set -Eeuo pipefail
module reset
module load python/3.11
PROJECT_STORAGE="${PROJECT:-${HOME}}"
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
export HF_DATASETS_OFFLINE=1
cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
M=icm.experiments.q10_pipeline_hypotheses
python -u -m $M.h7_audit
