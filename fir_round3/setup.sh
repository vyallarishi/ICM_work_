#!/usr/bin/env bash
# Fir, once, on a login node: verify the environment from the earlier wave, fetch and unpack the round-3 data.
# Usage:  bash fir_round3/setup.sh            (expects $SCRATCH/round3.tar.zst; prints the scp line if absent)
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); [ "$(dirname "$H")" = "$HOME/multilingual_icm/ICM_work_" ] || { echo "repo must be at ~/multilingual_icm/ICM_work_"; exit 1; }
D=$HOME/multilingual_icm/data/native_vs_sft/fir_round3; S=$SCRATCH/multilingual-icm; T=$SCRATCH/round3.tar.zst
for v in multilingual-icm sm-icm; do [ -x ${PROJECT:-$HOME}/venvs/$v/bin/python ] || { echo "missing venv $v: run fir_runs/setup_runs.sh first"; exit 1; }; done
for m in Qwen3-14B-Base Mistral-Nemo-Base-2407 gemma-3-12b-pt-text; do [ -f $S/assets/models/$m/config.json ] || echo "WARNING: model $m not found under $S/assets/models (needed for its runs)"; done
for f in sft_gold_filtered.jsonl eval_pipeline.jsonl eval_mgsm_human.jsonl; do [ -s $HOME/multilingual_icm/data/findings/runs_core24/$f ] || { echo "missing runs_core24/$f: run fir_runs/setup_runs.sh first"; exit 1; }; done
[ -s $T ] || { echo "fetch the data first (from Fir):  scp rvyalla@rorqual.alliancecan.ca:/scratch/rvyalla/fir_round3/round3.tar.zst $T"; exit 1; }
( cd $SCRATCH && sha256sum -c $H/transfer/round3.sha256 )
mkdir -p $D $S/assets/datasets; zstd -dc $T | tar -C $D -xf -
( cd $D && sha256sum -c --quiet $H/transfer/files.sha256 ) && echo "round-3 files verified"
[ -d $S/assets/datasets/gsm8k-106 ] || mv $D/gsm8k-106 $S/assets/datasets/gsm8k-106
# runs_core24/sft_translated.jsonl was never shipped in the earlier wave: it is in this tarball (verified above)
C24=$HOME/multilingual_icm/data/findings/runs_core24
[ -s $C24/sft_translated.jsonl ] || cp $D/runs_core24/sft_translated.jsonl $C24/sft_translated.jsonl
echo "af510e85f49d64d10222f4f6b8faa2453dc6f1d9aece036dbd737f1303a41d66  $C24/sft_translated.jsonl" | sha256sum -c
mkdir -p $SCRATCH/cache/{vllm,triton,xdg,hf} $D/logs $D/results
echo "setup done. Next:  STAGE=A bash $H/submit.sh   (dry run), then SUBMIT=1 STAGE=A bash $H/submit.sh"
