#!/usr/bin/env bash
# Q09 helper (CPU): text-only copy of a vision+text checkpoint (Gemma 3), so the student is
# trained and evaluated like Qwen and Mistral. Tensors are copied, not changed. -> def-zhijing
#     SRC=gemma-3-12b-pt sbatch extract_text_model.sbatch   (writes ${SRC}-text next to it)
#SBATCH --account=def-zhijing
#SBATCH --job-name=q09-extract-text
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/ICM_work_/logs/%x-%j.out
module reset
module load python/3.11
set -Eeuo pipefail
: "${SRC:?set SRC=<dir name under assets/models>}"
PROJECT_STORAGE="${PROJECT:-${HOME}}"
M=/scratch/rvyalla/multilingual-icm/assets/models
export PYTHONPATH="/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
source "${PROJECT_STORAGE}/venvs/sm-icm/bin/activate"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
python -u -m icm.experiments.q09_does_the_dataset_beat_translation.extract_text_model \
    --src "$M/$SRC" --dst "$M/${SRC}-text"
