#!/usr/bin/env bash
# Download the two models into $MODELS_DIR (~40 GB) and build the text-only Gemma checkpoint.
# Gemma is gated: accept the licence at https://huggingface.co/google/gemma-3-12b-pt and run
# `huggingface-cli login` (or set HF_TOKEN) first. Run on a node with internet access.
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/config.env"
source "$VENV_VLLM/bin/activate"
mkdir -p "$MODELS_DIR"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
hf download Qwen/Qwen3-14B-Base --local-dir "$MODELS_DIR/Qwen3-14B-Base"
hf download google/gemma-3-12b-pt --local-dir "$MODELS_DIR/gemma-3-12b-pt"
# Gemma 3 12B ships as a vision+text model; experiment 1 uses its text-only part, extracted exactly as
# in the original runs (tensors copied, vision tower dropped; the script verifies every key).
cd "$BUNDLE/code"
python -m icm.experiments.q09_does_the_dataset_beat_translation.extract_text_model \
    --src "$MODELS_DIR/gemma-3-12b-pt" --dst "$MODELS_DIR/gemma-3-12b-pt-text"
echo "models ready in $MODELS_DIR"
