#!/usr/bin/env bash
# Q09 step 2 (GPU, 4 x H100): LoRA fine-tuning on ONE arm. Submit once per arm:
#     ARM=translated sbatch train_lora.sbatch
#     ARM=method     sbatch train_lora.sbatch
#     ARM=method_fusion sbatch train_lora.sbatch
#     ARM=method_lenmatch sbatch train_lora.sbatch
# Everything except the data file is identical between the two runs.
# Smoke test first (20 steps):  ARM=method SMOKE=1 sbatch train_lora.sbatch
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=q09-train
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
: "${ARM:?set ARM=translated, method, method_fusion or method_lenmatch}"
source "${PROJECT_STORAGE}/venvs/sm-icm/bin/activate"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/scratch/rvyalla/multilingual-icm/results/q09/adapters/$ARM${SMOKE:+_smoke}
mkdir -p "$OUT"
torchrun --standalone --nproc_per_node 4 -m icm.experiments.q09_does_the_dataset_beat_translation.train_lora \
    --data q09_sft_$ARM.jsonl --out "$OUT" ${SMOKE:+--max-steps 20} \
    --match q09_sft_translated.jsonl --match q09_sft_method.jsonl --match q09_sft_method_fusion.jsonl \
    --match q09_sft_method_lenmatch.jsonl
