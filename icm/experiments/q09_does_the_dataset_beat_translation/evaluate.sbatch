#!/usr/bin/env bash
# Q09 step 3 (GPU, 1 x H100): base model + the 7 adapters (4 arms, 2 seeds for 3 of them)
# on the test split,
# identical prompts and greedy decoding. Smoke test: SMOKE=1 (10 prompts/language,
# reads the _smoke adapters).
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=q09-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=124G
#SBATCH --time=8:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
set +e
module reset >/dev/null 2>&1
module load python/3.11
module load cuda
set -e
set -Eeuo pipefail
PROJECT_STORAGE="${PROJECT:-${HOME}}"
cd /home/rvyalla/multilingual_icm/data/ICM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
A=/scratch/rvyalla/multilingual-icm/results/q09/adapters
S=${SMOKE:+_smoke}
ADAPTERS=""
for name in translated_s1 translated_s2 method_s1 method_s2 own_unlabelled_s1 own_unlabelled_s2 method_lenmatch_s1; do
    ADAPTERS="$ADAPTERS --adapter $name=$A/$name$S"
done
python -u -m icm.experiments.q09_does_the_dataset_beat_translation.evaluate $ADAPTERS \
    --out q09_results$S.json ${SMOKE:+--limit-per-lang 10}
