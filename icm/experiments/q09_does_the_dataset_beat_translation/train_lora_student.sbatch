#!/usr/bin/env bash
# Q09 step 2b (GPU, 4 x H100): the SAME LoRA fine-tuning on a DIFFERENT model family
# (student != generator), to separate data quality from self-distillation:
#     STUDENT=Mistral-Nemo-Base-2407 ARM=translated SEED=1 sbatch train_lora_student.sbatch
# arms: translated | method | own_unlabelled | method_lenmatch.  Everything except the
# data file and the seed is identical between runs; every arm trains on the same pairs.
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=q09-train-student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=256G
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
: "${ARM:?set ARM=translated, method, own_unlabelled or method_lenmatch}"
: "${SEED:?set SEED=1 or SEED=2}"
: "${STUDENT:?set STUDENT=<dir name under assets/models>}"
source "${PROJECT_STORAGE}/venvs/sm-icm/bin/activate"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/scratch/rvyalla/multilingual-icm/results/q09/adapters_${STUDENT}/${ARM}_s${SEED}${SMOKE:+_smoke}
mkdir -p "$OUT"
torchrun --standalone --nproc_per_node 4 -m icm.experiments.q09_does_the_dataset_beat_translation.train_lora \
    --model-dir /scratch/rvyalla/multilingual-icm/assets/models/$STUDENT --data q09_sft_$ARM.jsonl --out "$OUT" --seed $((20261002 + SEED)) ${SMOKE:+--max-steps 20} \
    --match q09_sft_translated.jsonl --match q09_sft_method.jsonl --match q09_sft_own_unlabelled.jsonl \
    --match q09_sft_method_lenmatch.jsonl
