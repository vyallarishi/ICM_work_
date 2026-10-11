# Sourced at the top of every job:  source "$BUNDLE/setup/job_env.sh" vllm|train
# Loads modules (if the cluster has them), activates the matching venv, points every cache at
# $CACHE_DIR (never /home), and sets LOCAL_TMP to node-local disk for staging models.
: "${BUNDLE:?BUNDLE not set: submit through the run scripts, which source config.env}"
_kind="${1:?usage: source job_env.sh vllm|train}"
if command -v module >/dev/null 2>&1; then
    module reset >/dev/null 2>&1 || true
    if [ "$_kind" = vllm ]; then _mods="$MODULES_VLLM"; else _mods="$MODULES_TRAIN"; fi
    [ -n "$_mods" ] && module load $_mods
fi
if [ "$_kind" = vllm ]; then source "$VENV_VLLM/bin/activate"; else source "$VENV_TRAIN/bin/activate"; fi
export LOCAL_TMP="${SLURM_TMPDIR:-/tmp/$USER/job_${SLURM_JOB_ID:-local}}"
mkdir -p "$LOCAL_TMP" "$CACHE_DIR"/{vllm,triton,xdg,hf,uv}
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm" TRITON_CACHE_DIR="$CACHE_DIR/triton" XDG_CACHE_HOME="$CACHE_DIR/xdg"
export HF_HOME="$CACHE_DIR/hf" UV_CACHE_DIR="$CACHE_DIR/uv"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$BUNDLE/code${PYTHONPATH:+:$PYTHONPATH}"
export ICM_RUN_DIR="$OUTPUT_DIR/run_dir"
# copy a model to node-local disk (reading weights straight from a parallel filesystem can crawl)
stage_model() {  # stage_model <model dir name>  -> prints the local path
    rsync -a "$MODELS_DIR/$1/" "$LOCAL_TMP/$1/" && echo "$LOCAL_TMP/$1"
}
echo "[job_env] $_kind | python $(command -v python) | LOCAL_TMP=$LOCAL_TMP | $(date)"
