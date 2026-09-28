#!/usr/bin/env bash
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=icm-qwen3-8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -Eeuo pipefail

module reset
module load python/3.11
module load cuda

: "${SCRATCH:?SCRATCH is not set}"
: "${SLURM_TMPDIR:?SLURM_TMPDIR is not set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
PROJECT_STORAGE="${PROJECT:-${HOME}}"
VENV_DIR="${VENV_DIR:-${PROJECT_STORAGE}/venvs/multilingual-icm}"
ASSET_ROOT="${ASSET_ROOT:-${SCRATCH}/multilingual-icm/assets}"
MODEL_SOURCE="${MODEL_DIR:-${ASSET_ROOT}/models/Qwen3-8B-Base}"
DATASET_SOURCE="${DATASET_DIR:-${ASSET_ROOT}/datasets/gsm8k-106}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b-native-6shot-main}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/configs/main_regimes.json}"
RESULT_ROOT="${RESULT_ROOT:-${SCRATCH}/multilingual-icm/results/${EXPERIMENT_NAME}}"
LANGUAGES="${LANGUAGES:-}"
LIMIT_PER_LANGUAGE="${LIMIT_PER_LANGUAGE:-}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/uv-cache}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf-cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"

[[ -d "${MODEL_SOURCE}" ]] || { echo "Missing model: ${MODEL_SOURCE}" >&2; exit 1; }
[[ -f "${DATASET_SOURCE}/offline_manifest.json" ]] || {
  echo "Missing offline dataset: ${DATASET_SOURCE}" >&2
  exit 1
}

mkdir -p "${SLURM_TMPDIR}/model" "${SLURM_TMPDIR}/data" "${RESULT_ROOT}/logs"
echo "[rorqual] Staging model and dataset to ${SLURM_TMPDIR}"
rsync -a "${MODEL_SOURCE}/" "${SLURM_TMPDIR}/model/"
rsync -a "${DATASET_SOURCE}/" "${SLURM_TMPDIR}/data/"

python scripts/preflight_offline.py \
  --model-dir "${SLURM_TMPDIR}/model" \
  --dataset-dir "${SLURM_TMPDIR}/data" \
  --language af

IFS=',' read -r -a GPU_IDS <<<"${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPU_IDS[@]}"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "No allocated GPUs found in CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

CPUS_PER_REPLICA=$(( SLURM_CPUS_PER_TASK / GPU_COUNT ))
SHARDS_ROOT="${RESULT_ROOT}/workers_${GPU_COUNT}gpu"
mkdir -p "${SHARDS_ROOT}"

echo "[rorqual] Launching ${GPU_COUNT} independent one-GPU vLLM replicas"
pids=()
for shard_index in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$shard_index]}"
  shard_dir="${SHARDS_ROOT}/shard_${shard_index}"
  worker_log="${RESULT_ROOT}/logs/shard-${shard_index}-${SLURM_JOB_ID}.log"
  mkdir -p "${shard_dir}"

  worker_cmd=(
    python scripts/run_experiment.py
    --manifest "${MANIFEST}"
    --model "${SLURM_TMPDIR}/model"
    --dataset "${SLURM_TMPDIR}/data"
    --output-dir "${shard_dir}"
    --language-shard-count "${GPU_COUNT}"
    --language-shard-index "${shard_index}"
    --generate-only
  )
  if [[ -n "${LIMIT_PER_LANGUAGE}" ]]; then
    worker_cmd+=(--limit-per-language "${LIMIT_PER_LANGUAGE}")
  fi
  if [[ -n "${LANGUAGES}" ]]; then
    # Intentional splitting: LANGUAGES is a space-separated list of language codes.
    # shellcheck disable=SC2206
    language_args=(${LANGUAGES})
    worker_cmd+=(--languages "${language_args[@]}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  OMP_NUM_THREADS="${CPUS_PER_REPLICA}" \
  MKL_NUM_THREADS="${CPUS_PER_REPLICA}" \
  "${worker_cmd[@]}" \
    >"${worker_log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one GPU worker failed. Check ${RESULT_ROOT}/logs/." >&2
  exit "${status}"
fi

echo "[rorqual] Pooling regimes, computing statistics, and building candidate views"
python scripts/merge_experiment_shards.py \
  --shards-root "${SHARDS_ROOT}" \
  --output-dir "${RESULT_ROOT}/final" \
  --overwrite

echo "[rorqual] Complete: ${RESULT_ROOT}/final"
