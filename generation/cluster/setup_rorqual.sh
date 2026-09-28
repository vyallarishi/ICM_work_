#!/usr/bin/env bash
set -Eeuo pipefail

# Run this script on a Rorqual login node, where outbound network access is available.
module reset
module load python/3.11
module load cuda

: "${SCRATCH:?SCRATCH is not set. Run this script on an Alliance login node.}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_STORAGE="${PROJECT:-${HOME}}"
VENV_DIR="${VENV_DIR:-${PROJECT_STORAGE}/venvs/multilingual-icm}"
ASSET_ROOT="${ASSET_ROOT:-${SCRATCH}/multilingual-icm/assets}"
MODEL_DIR="${MODEL_DIR:-${ASSET_ROOT}/models/Qwen3-8B-Base}"
DATASET_DIR="${DATASET_DIR:-${ASSET_ROOT}/datasets/gsm8k-106}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/uv-cache}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf-cache}"
export HF_XET_HIGH_PERFORMANCE=1
export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${ASSET_ROOT}" "${REPO_ROOT}/logs"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup_rorqual] Installing uv into ${HOME}/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "[setup_rorqual] Creating environment at ${VENV_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv --python "$(command -v python)" "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
uv sync --active --project "${REPO_ROOT}"

echo "[setup_rorqual] Downloading offline assets under ${ASSET_ROOT}"
python "${REPO_ROOT}/scripts/download_assets.py" \
  --model Qwen/Qwen3-8B-Base \
  --dataset open-multilang/gsm8k-106 \
  --model-dir "${MODEL_DIR}" \
  --dataset-dir "${DATASET_DIR}"

python -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)'

cat <<EOF
[setup_rorqual] Ready.
VENV_DIR=${VENV_DIR}
MODEL_DIR=${MODEL_DIR}
DATASET_DIR=${DATASET_DIR}

Submit from the repository root with:
  sbatch cluster/rorqual_generate.sbatch
EOF
