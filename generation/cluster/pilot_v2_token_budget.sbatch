#!/usr/bin/env bash
# PILOT v2: the languages the first pilot flagged, re-generated with a 1,536-token
# budget and an 8,192-token context (configs/best_regimes_v3_1536tok.json), plus candidate
# replacements. The context matters: the generator skips any prompt longer than
# max_model_len - max_new_tokens, and a 2-shot Burmese prompt is ~3,000 tokens.
# Burmese lost 32% of its chains to the 768-token limit, Telugu 16%, Gujarati 13%.
# Override the list at submit time:  LANGUAGES="my te gu so ig" sbatch cluster/pilot_v2_token_budget.sbatch
# Submit from this repo's generation/ directory. Afterwards (CPU, seconds):
#   PYTHONPATH=.. python ../tools/pilot_report.py --root $SCRATCH/multilingual-icm/results/qwen3-14b-pilot-v3 \
#       --jury-root $SCRATCH/multilingual-icm/results/qwen3-14b-full37
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=gen-14b-pilot-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=256G
#SBATCH --time=1:30:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-14b-pilot-v3}"
export MODEL_DIR="${SCRATCH}/multilingual-icm/assets/models/Qwen3-14B-Base"
export MANIFEST="${SLURM_SUBMIT_DIR}/configs/best_regimes_v3_1536tok.json"
export LIMIT_PER_LANGUAGE=150
# default: the three token-limited languages + four replacement candidates for Hausa / Yoruba
export LANGUAGES="${LANGUAGES:-my te gu ta so ig pa si}"
source "${SLURM_SUBMIT_DIR}/cluster/rorqual_generate.sbatch"
