#!/usr/bin/env bash
# Fir: all remaining core-24 trainings (15, 2 GPUs each) and their evaluations (4 jobs, 1 GPU each), on the
# Rorqual job scripts unchanged (copied to ~/multilingual_icm/data/findings/jobs by setup_runs.sh).
#   bash submit_runs.sh            dry run: sbatch --test-only for each job type, nothing submitted
#   SUBMIT=1 bash submit_runs.sh   submit: prepare -> 15 trainings -> 4 evaluations (dependencies wired)
# Resources fit Fir's nodes via: ACCOUNT (def-zhijing) GPU_TYPE (h100) TRAIN_CPUS (16) TRAIN_MEM (128G)
# EVAL_CPUS (12) EVAL_MEM (120G). These never change results (the recipe is fixed inside the job scripts).
set -euo pipefail
ACC=${ACCOUNT:-def-zhijing}; GPU=${GPU_TYPE:-h100}
H=$(cd "$(dirname "$0")" && pwd); J=$HOME/multilingual_icm/data/findings/jobs
TR=(--account=$ACC --gpus-per-node=$GPU:2 --cpus-per-task=${TRAIN_CPUS:-16} --mem=${TRAIN_MEM:-128G} --time=12:00:00)
EV=(--account=$ACC --gpus-per-node=$GPU:1 --cpus-per-task=${EVAL_CPUS:-12} --mem=${EVAL_MEM:-120G} --time=12:00:00)
Q=Qwen3-14B-Base; M=Mistral-Nemo-Base-2407; G=gemma-3-12b-pt-text
if [ "${SUBMIT:-0}" != 1 ]; then
    printf "prepare  "; sbatch --test-only --account=$ACC $H/prepare_runs.sbatch 2>&1 | grep -o "to start.*partition [a-z_0-9]*\|rror.*"
    printf "train    "; STUDENT=$Q ARM=method SEED=1 NGPU=2 sbatch --test-only "${TR[@]}" $J/core24_train.sbatch 2>&1 | grep -o "to start.*partition [a-z_0-9]*\|rror.*"
    printf "evaluate "; STUDENT=$Q ADAPTERS=x TAG=x sbatch --test-only "${EV[@]}" $J/core24_eval.sbatch 2>&1 | grep -o "to start.*partition [a-z_0-9]*\|rror.*"
    echo "dry run only; nothing submitted. Submit with: SUBMIT=1 bash $0"; exit 0
fi
P=$(sbatch --parsable --account=$ACC $H/prepare_runs.sbatch); echo "prepare $P"
train() { STUDENT=$1 ARM=$2 SEED=$3 NGPU=2 sbatch --parsable --dependency=afterok:$P "${TR[@]}" $J/core24_train.sbatch; }
evaluate() { STUDENT=$1 ADAPTERS="$2" TAG=$3 sbatch --parsable --dependency=afterok$4 "${EV[@]}" $J/core24_eval.sbatch; }
d=""; for a in "gold_filtered 2" "method 1" "method 2" "method_lenmatch 1"; do set -- $a; i=$(train $M $1 $2); d="$d:$i"; echo "train $M $1 s$2  $i"; done
echo "eval  $M  $(evaluate $M "gold_filtered_s2 method_s1 method_s2 method_lenmatch_s1" fir $d)"
d=""; for a in "method 1" "method 2" "method_lenmatch 1"; do set -- $a; i=$(train $Q $1 $2); d="$d:$i"; echo "train $Q $1 s$2  $i"; done
echo "eval  $Q  $(evaluate $Q "method_s1 method_s2 method_lenmatch_s1" wave3 $d)"
d=""; for a in "own_unlabelled 1" "own_unlabelled 2" "gold_filtered 1" "gold_filtered 2"; do set -- $a; i=$(train $G $1 $2); d="$d:$i"; echo "train $G $1 s$2  $i"; done
echo "eval  $G wave2  $(evaluate $G "own_unlabelled_s1 own_unlabelled_s2 gold_filtered_s1 gold_filtered_s2" wave2 $d)"
d=""; for a in "method 1" "method 2" "method_lenmatch 1"; do set -- $a; i=$(train $G $1 $2); d="$d:$i"; echo "train $G $1 s$2  $i"; done
echo "eval  $G wave3  $(evaluate $G "method_s1 method_s2 method_lenmatch_s1" wave3 $d)"
echo "submitted: 1 prepare, 14 trainings, 4 evaluations. Watch: squeue -u $USER"
