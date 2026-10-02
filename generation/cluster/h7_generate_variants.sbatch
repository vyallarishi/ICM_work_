#!/usr/bin/env bash
# H7 step 2 (GPU, 4 x H100): generate chains for the 500 variants in 37 languages,
# with EXACTLY the record's generator and regimes (Qwen3-14B-Base, best_regimes_v2:
# 8+8+4 chains, 768 new tokens, 2-shot prompts), so variants and originals differ
# only in the numbers. Submit from this repo's generation/ directory:
#     cd ICM_work_/generation && sbatch cluster/h7_generate_variants.sbatch
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=h7-gen-variants
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
export EXPERIMENT_NAME="qwen3-14b-variants500"
export MODEL_DIR="${SCRATCH}/multilingual-icm/assets/models/Qwen3-14B-Base"
export DATASET_DIR="${SCRATCH}/multilingual-icm/assets/datasets/gsm8k-106-variants500"
export MANIFEST="${SLURM_SUBMIT_DIR}/configs/best_regimes_v2.json"
export LANGUAGES="af am ar az bg bn ceb cs de el en es eu fa fr hi id is it kk kn ko ky ml mr ms nl pl pt ru sw th tr uk ur vi zh-tw"
source "${SLURM_SUBMIT_DIR}/cluster/rorqual_generate.sbatch"
