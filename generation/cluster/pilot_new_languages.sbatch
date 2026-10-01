#!/usr/bin/env bash
# PILOT for the 50-language scale-up: 13 new languages x 150 problems with the
# exact generator and regimes of the 37-language record (Qwen3-14B-Base,
# best_regimes_v2: 8+8+4 chains at T=0.2/0.5/0.8, 768 new tokens).
# Submit from this repo's generation/ directory:
#     cd ICM_work_/generation && sbatch cluster/pilot_new_languages.sbatch
# Afterwards (CPU, seconds):
#     PYTHONPATH=.. python ../tools/pilot_report.py --root $SCRATCH/multilingual-icm/results/qwen3-14b-pilot13
# Reference: 22 languages x 150 took 30 min on 4 x H100 (job 19126878).
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=gen-14b-pilot13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=256G
#SBATCH --time=1:30:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
export EXPERIMENT_NAME="qwen3-14b-pilot13"
export MODEL_DIR="${SCRATCH}/multilingual-icm/assets/models/Qwen3-14B-Base"
export MANIFEST="${SLURM_SUBMIT_DIR}/configs/best_regimes_v2.json"
export LIMIT_PER_LANGUAGE=150
# high: ja zh-cn sv hu | mid: he ta te gu ne ka | low: ha yo my   (ja, zh-cn, te complete MGSM)
export LANGUAGES="ja zh-cn sv hu he ta te gu ne ka ha yo my"
source "${SLURM_SUBMIT_DIR}/cluster/rorqual_generate.sbatch"
