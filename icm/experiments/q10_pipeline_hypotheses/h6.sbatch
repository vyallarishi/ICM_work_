#!/usr/bin/env bash
# H6 (translationese): third-script letters in translated vs native solutions.
# CPU. Needs the q09 training sets (build_sft_sets.sbatch). -> def-zhijing
#SBATCH --account=def-zhijing
#SBATCH --job-name=h6-translationese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
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
echo "===== H6"; python -u -m $M.h6_translationese
