#!/usr/bin/env bash
# Any Alliance cluster (Fir, Nibi, ...), run ONCE on a LOGIN node (needs internet).
# STUDENTS="<space-separated>" limits setup to those students (default: all three); each cluster sets up only its own.
# Builds the training environment from PyPI with uv, unpacks the core-24 training/test files (shipped in
# fir_runs/transfer/), and downloads Mistral-Nemo-Base-2407 and gemma-3-12b-pt. Both are GATED on Hugging Face:
# log in first (`hf auth login`) with an account that has accepted both licences.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
[ "$(dirname "$H")" = "$HOME/multilingual_icm/ICM_work_" ] || { echo "repo must be at ~/multilingual_icm/ICM_work_"; exit 1; }
F=$HOME/multilingual_icm/data/findings; S=/scratch/$USER/multilingual-icm
module load python/3.11 cuda arrow
VV=${PROJECT:-$HOME}/venvs/multilingual-icm; VT=${PROJECT:-$HOME}/venvs/sm-icm
STUDENTS="${STUDENTS:-Qwen3-14B-Base Mistral-Nemo-Base-2407 gemma-3-12b-pt-text}"; echo "students on this cluster: $STUDENTS"
if [ ! -x $VV/bin/python ]; then   # the vLLM environment (evaluation), same versions as Rorqual
    python3.11 -m venv $VV && $VV/bin/pip install -q uv
    $VV/bin/uv pip install --python $VV/bin/python vllm==0.25.1 datasets==5.0.0 rich==15.0.0
fi
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
for st in $STUDENTS; do
    # pinned to the exact revisions Rorqual downloaded (from its download records), so files match byte for byte
    case $st in Qwen3-14B-Base) m=Qwen/Qwen3-14B-Base; r=0b0bd3732e2c374d483664439ea334928b65f304;;
                Mistral-Nemo-Base-2407) m=mistralai/Mistral-Nemo-Base-2407; r=a4477a2f977929a969745b69bbd62e03043551a5;;
                gemma-3-12b-pt-text) m=google/gemma-3-12b-pt; r=295efb63d01a7017928f273a94ebb86105c9526f;;
                *) echo "unknown student $st"; exit 1;; esac
    [ -f $S/assets/models/${m#*/}/config.json ] || $VV/bin/hf download $m --revision $r --local-dir $S/assets/models/${m#*/}
done
echo "setup done; next: STUDENTS=\"$STUDENTS\" bash $H/submit_runs.sh (dry run)"
