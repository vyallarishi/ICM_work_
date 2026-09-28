#!/usr/bin/env bash
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=icm-retry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -Eeuo pipefail

module reset
module load python/3.11
module load cuda

: "${SCRATCH:?SCRATCH is not set}"
: "${SLURM_TMPDIR:?SLURM_TMPDIR is not set}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
PROJECT_STORAGE="${PROJECT:-${HOME}}"
VENV_DIR="${VENV_DIR:-${PROJECT_STORAGE}/venvs/multilingual-icm}"
ASSET_ROOT="${ASSET_ROOT:-${SCRATCH}/multilingual-icm/assets}"
MODEL_SOURCE="${MODEL_DIR:-${ASSET_ROOT}/models/Qwen3-8B-Base}"
DATASET_SOURCE="${DATASET_DIR:-${ASSET_ROOT}/datasets/gsm8k-106}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b-native-6shot-main}"
BASE_DIR="${BASE_DIR:-${SCRATCH}/multilingual-icm/results/${EXPERIMENT_NAME}/final}"
RETRY_MANIFEST="${RETRY_MANIFEST:-${REPO_ROOT}/configs/retry_regimes.json}"

export HF_HOME="${HF_HOME:-${SCRATCH}/hf-cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"

[[ -d "${BASE_DIR}/analysis/item_candidate_stats" ]] || {
  echo "Missing candidate analysis: ${BASE_DIR}" >&2
  exit 1
}
mkdir -p "${SLURM_TMPDIR}/model" "${SLURM_TMPDIR}/data"
rsync -a "${MODEL_SOURCE}/" "${SLURM_TMPDIR}/model/"
rsync -a "${DATASET_SOURCE}/" "${SLURM_TMPDIR}/data/"

python scripts/run_adaptive_retry.py \
  --base-dir "${BASE_DIR}" \
  --manifest "${RETRY_MANIFEST}" \
  --model "${SLURM_TMPDIR}/model" \
  --dataset "${SLURM_TMPDIR}/data" \
  --overwrite

echo "[rorqual] Adaptive retry complete: ${BASE_DIR}"
