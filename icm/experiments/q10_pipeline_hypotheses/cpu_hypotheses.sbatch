#!/usr/bin/env bash
# H1 (v2 rules), H2 (calibrated confidence), H5 (jury size / chains per pool).
# CPU, independent of everything else; each writes runs/h*.json. -> def-zhijing
#SBATCH --account=def-zhijing
#SBATCH --job-name=q10-cpu-hypotheses
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
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
echo "===== H1"; python -u -m $M.h1_rules
echo "===== H2"; python -u -m $M.h2_confidence
echo "===== H5"; python -u -m $M.h5_scaling
