#!/usr/bin/env bash
# Fir, run ONCE on a LOGIN node (needs internet). The Qwen record ships with this branch (fir_labelling/transfer/).
# Recreates Rorqual's folder layout (same username, same absolute paths) so the labelling scripts run unchanged,
# builds the Python environment from PyPI with uv, and downloads Qwen3-14B-Base.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
[ "$(dirname "$H")" = "$HOME/multilingual_icm/ICM_work_" ] || { echo "clone this repo to ~/multilingual_icm/ICM_work_ (the scripts use absolute paths)"; exit 1; }
# the record comes with this branch (transfer/, split into <100 MB parts for GitHub); rebuild it on /scratch
T=/scratch/$USER/fir_transfer
if [ ! -f $T/transfer.sha256 ]; then
    mkdir -p $T
    cat $H/transfer/record_qwen3-14b-full66-7473.tar.zst.part* > $T/record_qwen3-14b-full66-7473.tar.zst
    cp $H/transfer/*.sha256 $T/
fi
( cd $T && sha256sum -c transfer.sha256 ) || { echo "record file damaged: re-clone the branch"; exit 1; }
F=$HOME/multilingual_icm/data/findings
mkdir -p $F/jobs $F/results $F/logs $F/runs_all67 $HOME/multilingual_icm/data/ICM_work_/runs
cp $H/jobs/* $F/jobs/; cp $H/data/language_selection.json $F/results/
S=/scratch/$USER/multilingual-icm; mkdir -p $S/results $S/assets/models /scratch/$USER/cache/hf
module load python/3.11 cuda arrow
VENV=${PROJECT:-$HOME}/venvs/multilingual-icm
if [ ! -x $VENV/bin/python ]; then
    python3.11 -m venv $VENV
    $VENV/bin/pip install -q uv
    $VENV/bin/uv pip install --python $VENV/bin/python vllm==0.25.1 datasets==5.0.0 rich==15.0.0
fi
$VENV/bin/python -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'))"
HF_HOME=/scratch/$USER/cache/hf $VENV/bin/hf download Qwen/Qwen3-14B-Base --local-dir $S/assets/models/Qwen3-14B-Base
echo "setup done; next: bash ~/multilingual_icm/ICM_work_/fir_labelling/submit_fir.sh   (dry run), then with SUBMIT=1"
