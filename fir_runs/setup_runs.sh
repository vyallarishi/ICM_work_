#!/usr/bin/env bash
# Fir, run ONCE on a LOGIN node (needs internet), AFTER fir_labelling/setup_fir.sh has been run.
# Builds the training environment from PyPI with uv, unpacks the core-24 training/test files (shipped in
# fir_runs/transfer/), and downloads Mistral-Nemo-Base-2407 and gemma-3-12b-pt. Both are GATED on Hugging Face:
# log in first (`hf auth login`) with an account that has accepted both licences.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
[ "$(dirname "$H")" = "$HOME/multilingual_icm/ICM_work_" ] || { echo "repo must be at ~/multilingual_icm/ICM_work_"; exit 1; }
F=$HOME/multilingual_icm/data/findings; S=/scratch/$USER/multilingual-icm
module load python/3.11 cuda arrow
VV=${PROJECT:-$HOME}/venvs/multilingual-icm; VT=${PROJECT:-$HOME}/venvs/sm-icm
[ -x $VV/bin/hf ] || { echo "run fir_labelling/setup_fir.sh first"; exit 1; }
export HF_HOME=/scratch/$USER/cache/hf
$VV/bin/hf auth whoami >/dev/null 2>&1 || { echo "NOT LOGGED IN to Hugging Face. Run:  HF_HOME=/scratch/$USER/cache/hf $VV/bin/hf auth login   (token from huggingface.co/settings/tokens)"; exit 1; }
mkdir -p $F/jobs $F/results $F/logs $HOME/multilingual_icm/data/ICM_work_/runs /scratch/$USER/cache/{vllm,triton,xdg,hf} $S/assets/models
cp $H/jobs/* $F/jobs/
# 1. training environment (the versions of every training run so far)
if [ ! -x $VT/bin/python ]; then
    python3.11 -m venv $VT && $VT/bin/pip install -q uv
    $VT/bin/uv pip install --python $VT/bin/python torch==2.13.0 transformers==4.57.6 peft==0.19.1 accelerate==1.14.0 safetensors==0.8.0
fi
$VT/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ('torch','transformers','peft','accelerate')})"
# 2. training + test files
T=/scratch/$USER/fir_runs; mkdir -p $T
cat $H/transfer/runs_core24.tar.zst.part* > $T/runs_core24.tar.zst; cp $H/transfer/*.sha256 $T/
( cd $T && sha256sum -c transfer.sha256 )
zstd -dc $T/runs_core24.tar.zst | tar -C $F -xf -
( cd $F/runs_core24 && sha256sum -c --quiet $T/runs_core24.sha256 ) && echo "training/test files: 7 verified"
# 3. models (checked against Rorqual's checksums by the prepare job)
for m in mistralai/Mistral-Nemo-Base-2407 google/gemma-3-12b-pt; do
    $VV/bin/hf download $m --local-dir $S/assets/models/${m#*/}
done
echo "setup done; next: bash $H/submit_runs.sh (dry run)"
