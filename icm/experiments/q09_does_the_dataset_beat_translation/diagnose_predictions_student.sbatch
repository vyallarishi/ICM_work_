#!/usr/bin/env bash
# Q09 step 4b (GPU, 1 x H100): the format check (diagnose_predictions.py) for the other-family
# student: its base model + 6 adapters, every prediction saved, strict vs format-blind scoring.
# Accuracies must reproduce q09_results_${STUDENT}.json.
#     STUDENT=Mistral-Nemo-Base-2407 sbatch diagnose_predictions_student.sbatch
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=q09-diagnose-student
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
: "${STUDENT:?set STUDENT=<dir name under assets/models>}"
A=/scratch/rvyalla/multilingual-icm/results/q09/adapters_${STUDENT}
S=${SMOKE:+_smoke}
ADAPTERS=""
for name in translated_s1 translated_s2 method_s1 method_s2 own_unlabelled_s1 own_unlabelled_s2; do
    ADAPTERS="$ADAPTERS --adapter $name=$A/$name$S"
done
python -u -m icm.experiments.q09_does_the_dataset_beat_translation.diagnose_predictions $ADAPTERS \
    --model-dir /scratch/rvyalla/multilingual-icm/assets/models/$STUDENT \
    --tag "_${STUDENT}$S" ${SMOKE:+--limit-per-lang 10}
