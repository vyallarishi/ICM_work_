#!/usr/bin/env bash
set -Eeuo pipefail

: "${HF_TOKEN:=}"
: "${HF_RESULTS_REPO:=}"
: "${HF_RESULTS_PATH:=}"
: "${HF_PRIVATE:=0}"
: "${HF_UPLOAD:=0}"
: "${HF_UPLOAD_EACH_LANGUAGE:=1}"

: "${MODEL:=Qwen/Qwen3-8B-Base}"
: "${DATASET:=open-multilang/gsm8k-106}"
: "${SPLIT:=train}"
: "${PROMPT_TEMPLATE:=zero_shot_cot}"
: "${ENGLISH_PROMPT_ONLY:=0}"
: "${ASSISTANT_PREAMBLE_FILE:=}"
: "${FEW_SHOT_COUNT:=0}"
: "${OUTPUT_DIR:=}"

: "${DTYPE:=bfloat16}"
: "${TENSOR_PARALLEL_SIZE:=auto}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_NUM_SEQS:=}"
: "${MAX_MODEL_LEN:=4096}"
: "${MAX_NEW_TOKENS:=768}"
: "${BATCH_SIZE:=128}"
: "${TEMPERATURE:=0.7}"
: "${TOP_P:=0.95}"
: "${N:=12}"
: "${SEED:=0}"
: "${TOLERANCE:=0}"

: "${LANGUAGES:=}"
: "${LANGUAGE_SHARD_COUNT:=1}"
: "${LANGUAGE_SHARD_INDEX:=0}"
: "${LIMIT_PER_LANGUAGE:=}"
: "${QUANTIZATION:=}"
: "${TRUST_REMOTE_CODE:=0}"
: "${OVERWRITE:=0}"
: "${MANIFEST:=configs/main_regimes.json}"
: "${RETRY_MANIFEST:=configs/retry_regimes.json}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash jobrun.sh [--setup-only] [--smoke] [--generate] [--views] [--retry] [--all] [--overwrite]

Modes:
  --setup-only   Install/sync dependencies, then stop.
  --smoke        Generate a tiny af/en run, then build candidate views.
  --generate     Run manifest generation, merge, analysis, and candidate views. Default.
  --views        Rebuild analysis and candidate views from OUTPUT_DIR/pooled.
  --retry        Run adaptive retry from OUTPUT_DIR.
  --all          Same as --generate.
  --overwrite    Rewrite existing generated and derived files.

Configure with environment variables, a local .env file, or edits at the top of this script.
EOF
}

log() {
  printf '[jobrun] %s\n' "$*"
}

die() {
  printf '[jobrun] ERROR: %s\n' "$*" >&2
  exit 1
}

load_env_file() {
  local env_file="${ENV_FILE:-${PROJECT_ROOT}/.env}"
  if [[ -f "${env_file}" ]]; then
    log "Loading ${env_file}"
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

install_uv_if_needed() {
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    log "uv already installed: $(uv --version)"
    return
  fi
  log "Installing uv"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    die "Install curl or wget first, then rerun this script."
  fi
}

detect_visible_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    local visible="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -n "${visible}" ]]; then
      awk -F',' '{print NF}' <<<"${visible}"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l | tr -d ' '
    return
  fi
  printf '1\n'
}

MODE="generate"
SETUP_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --setup-only)
      SETUP_ONLY=1
      ;;
    --smoke)
      MODE="smoke"
      ;;
    --generate)
      MODE="generate"
      ;;
    --views)
      MODE="views"
      ;;
    --retry)
      MODE="retry"
      ;;
    --all)
      MODE="all"
      ;;
    --overwrite)
      OVERWRITE=1
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

load_env_file
install_uv_if_needed

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
[[ -n "${HF_TOKEN}" ]] && export HF_TOKEN
mkdir -p "${HF_HOME}"

log "Syncing Python environment"
uv sync

if [[ "${SETUP_ONLY}" == "1" ]]; then
  log "Setup complete"
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-results/smoke_qwen3_8b_base_train_chains}"
  LANGUAGES="${LANGUAGES:-af en}"
  LIMIT_PER_LANGUAGE="${LIMIT_PER_LANGUAGE:-2}"
  if [[ "${MANIFEST}" == "configs/main_regimes.json" ]]; then
    MANIFEST="configs/pilot_regimes.json"
  fi
else
  if [[ "${LANGUAGE_SHARD_COUNT}" -gt 1 ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3_8b_base_train_chains_shard_${LANGUAGE_SHARD_INDEX}_of_${LANGUAGE_SHARD_COUNT}}"
  else
    OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3_8b_base_train_chains}"
  fi
fi

if [[ "${TENSOR_PARALLEL_SIZE}" == "auto" ]]; then
  TENSOR_PARALLEL_SIZE="$(detect_visible_gpu_count)"
fi

run_generate() {
  local cmd=(
    uv run python scripts/run_experiment.py
    --manifest "${MANIFEST}"
    --output-dir "${OUTPUT_DIR}"
    --language-shard-count "${LANGUAGE_SHARD_COUNT}"
    --language-shard-index "${LANGUAGE_SHARD_INDEX}"
  )

  [[ "${OVERWRITE}" == "1" ]] && cmd+=(--overwrite)
  [[ -n "${LIMIT_PER_LANGUAGE}" ]] && cmd+=(--limit-per-language "${LIMIT_PER_LANGUAGE}")

  if [[ -n "${LANGUAGES}" ]]; then
    # shellcheck disable=SC2206
    local language_args=(${LANGUAGES})
    cmd+=(--languages "${language_args[@]}")
  fi

  if [[ "${HF_UPLOAD}" == "1" && -n "${HF_RESULTS_REPO}" ]]; then
    cmd+=(--hf-results-repo "${HF_RESULTS_REPO}")
    [[ -n "${HF_RESULTS_PATH}" ]] && cmd+=(--hf-results-path "${HF_RESULTS_PATH}")
    [[ "${HF_PRIVATE}" == "1" ]] && cmd+=(--hf-private)
    [[ "${HF_UPLOAD_EACH_LANGUAGE}" != "1" ]] && cmd+=(--no-hf-upload-each-language)
  fi

  printf '[jobrun] Running:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

run_views() {
  local cmd=(
    uv run python scripts/build_candidate_views.py
    --input-dir "${OUTPUT_DIR}/pooled"
    --output-dir "${OUTPUT_DIR}"
  )
  [[ "${OVERWRITE}" == "1" ]] && cmd+=(--overwrite)
  printf '[jobrun] Running:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

run_retry() {
  local cmd=(
    uv run python scripts/run_adaptive_retry.py
    --base-dir "${OUTPUT_DIR}"
    --manifest "${RETRY_MANIFEST}"
  )
  [[ "${OVERWRITE}" == "1" ]] && cmd+=(--overwrite)
  printf '[jobrun] Running:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

case "${MODE}" in
  smoke|all)
    run_generate
    ;;
  generate)
    run_generate
    ;;
  views)
    run_views
    ;;
  retry)
    run_retry
    ;;
esac
