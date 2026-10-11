#!/usr/bin/env bash
# Checks everything needed before submitting anything. Light: no GPU, no model loading.
#   bash setup/check_setup.sh
# Prints PASS/FAIL per item and ends with a summary; fix every FAIL before running experiments.
source "$(cd "$(dirname "$0")/.." && pwd)/config.env"
fail=0
ok()  { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; fail=$((fail + 1)); }
for v in BUNDLE ACCOUNT_GPU ACCOUNT_CPU MODELS_DIR OUTPUT_DIR CACHE_DIR VENV_VLLM VENV_TRAIN GPU_TYPE; do
    [ -n "${!v:-}" ] && ok "config $v=${!v}" || bad "config $v is empty"
done
n=$(ls -d "$BUNDLE"/data/gsm8k-106/{train,test}/*/ 2>/dev/null | wc -l)
[ "$n" -eq 134 ] && ok "dataset present (134 tables)" || bad "dataset has $n of 134 tables (run setup/build_data.py download, then build)"
( cd "$BUNDLE" && sha256sum -c --quiet setup/fingerprints/derived_files.sha256 ) 2>/dev/null && ok "training and test files match the original cluster" || bad "training/test files missing or different (run setup/build_data.py build)"
for m in Qwen3-14B-Base gemma-3-12b-pt-text; do
    [ -f "$MODELS_DIR/$m/config.json" ] && [ -n "$(ls "$MODELS_DIR/$m"/*.safetensors 2>/dev/null)" ] \
        && ok "model $m" || bad "model $m missing in $MODELS_DIR (run setup/download_models.sh)"
done
check_env() {  # check_env <venv> <pinned requirements file>
    [ -x "$1/bin/python" ] || { bad "venv $1 missing"; return; }
    "$1/bin/python" - "$2" <<'PY' && ok "venv $1 has the pinned versions" || bad "venv $1: version mismatch (see above)"
import sys, importlib.metadata as md
bad = 0
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if not line: continue
    name, want = line.split("==")
    try: have = md.version(name)
    except md.PackageNotFoundError: have = None
    if have is None or have.split("+")[0] != want:
        print(f"      {name}: have {have}, want {want}"); bad += 1
sys.exit(1 if bad else 0)
PY
}
check_env "$VENV_VLLM" "$BUNDLE/setup/requirements-vllm.txt"
check_env "$VENV_TRAIN" "$BUNDLE/setup/requirements-train.txt"
command -v sbatch >/dev/null && ok "sbatch available" || bad "sbatch not found"
for a in "$ACCOUNT_GPU:--gpus-per-node=$GPU_TYPE:1" "$ACCOUNT_CPU:"; do
    acc=${a%%:*}; g=${a#*:}
    sbatch --test-only --account="$acc" $g --time=0:10:00 --wrap=true >/dev/null 2>&1 \
        && ok "Slurm accepts account $acc ${g:+with $g}" || bad "Slurm rejects account $acc ${g:+with $g}"
done
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" 2>/dev/null && ok "output/cache dirs writable" || bad "cannot create $OUTPUT_DIR or $CACHE_DIR"
echo; [ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "$fail CHECK(S) FAILED"
exit $fail
