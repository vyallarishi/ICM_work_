#!/usr/bin/env bash
# Reproduce the ICM headline (55,485 pools, 99.23% coverage @ 94.45% precision)
# from the raw /scratch pools + the cached regeneration answers in data/ICM_work_/runs/.
# CPU only -> def-zhijing (rrg-zhijing rejects CPU jobs). No GPU, no model load.
#SBATCH --account=def-zhijing
#SBATCH --job-name=icm-reproduce-headline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
set -Eeuo pipefail
module reset
module load python/3.11
PROJECT_STORAGE="${PROJECT:-${HOME}}"
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
# Jobs run from the data runs/ dir: the scripts read caches and write results in the current
# directory, exactly as they did from ~/phase0.
cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"

echo "== 1. the clean method (icm/method) on the real data, vs the recorded headline"
python -u -m icm.method.run decide --out method_results.json
python -u /home/rvyalla/multilingual_icm/ICM_work_/tools/compare_results.py method_results.json /home/rvyalla/multilingual_icm/data/ICM_work_/recorded_results/stage25_results.json

echo "== 2. the original experiment code, re-run, vs the recorded headline"
python -u -m icm.experiments.q04_full_scale_headline.full_scale_37x1500 --phase decide
python -u /home/rvyalla/multilingual_icm/ICM_work_/tools/compare_results.py stage25_results.json /home/rvyalla/multilingual_icm/data/ICM_work_/recorded_results/stage25_results.json

echo "== 3. independent recomputation from raw files (own loops, no shared code)"
python -u -m icm.experiments.q04_full_scale_headline.verify_full_scale_independently

echo "== 4. audit recompute + gold-leak test on the 37x1500 root"
python -u -m icm.experiments.q07_audits.audit_stage25_recompute
python -u -m icm.experiments.q07_audits.audit_gold_leak_1500
