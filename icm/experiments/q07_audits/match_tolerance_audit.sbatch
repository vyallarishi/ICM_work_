#!/usr/bin/env bash
# Real-data audit of the 0.5% answer tolerance (tools/match_tolerance_audit.py)
# on the 37x1500 headline root and the 13-language pilot. CPU only -> def-zhijing.
#SBATCH --account=def-zhijing
#SBATCH --job-name=icm-match-tolerance-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
set -Eeuo pipefail
module reset
module load python/3.11
PROJECT_STORAGE="${PROJECT:-${HOME}}"
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
TOOLS=/home/rvyalla/multilingual_icm/ICM_work_/tools
R=/scratch/rvyalla/multilingual-icm/results

echo "== 37 x 1500 headline root"
python -u "$TOOLS/match_tolerance_audit.py" --root $R/qwen3-14b-full37-1500 --out match_tolerance_full37_1500.json
echo "== 13-language pilot"
python -u "$TOOLS/match_tolerance_audit.py" --root $R/qwen3-14b-pilot13 --out match_tolerance_pilot13.json
