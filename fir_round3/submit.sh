#!/usr/bin/env bash
# Round 3 on Fir. Stages (STAGE=...):
#   A   Tier 1: build controls (CPU) -> 7 arms x 3 students at seed 2 (19 trainings) -> 3 evaluations
#   B   Tier 2b: flaw-tolerance, 18 trainings (Qwen) -> 3 evaluations
#   C1  Tier 2a: Mistral generation SMOKE (50 problems/language); inspect gen_check.txt, then
#   C2  Tier 2a: full Mistral generation -> build arms (CPU) -> 10 trainings -> 2 evaluations
# Dry run by default (sbatch --test-only); SUBMIT=1 submits. ACCOUNT (def-zhijing), GPU_TYPE (h100) adjustable.
set -euo pipefail
R3=$HOME/multilingual_icm/ICM_work_/fir_round3; D=$HOME/multilingual_icm/data/native_vs_sft/fir_round3; RUNS=$D/runs; RES=$D/results
C24=$HOME/multilingual_icm/data/findings/runs_core24; AD=$SCRATCH/multilingual-icm/results/round3; mkdir -p $RES $D/logs
ACC=${ACCOUNT:-def-zhijing}; GPU=${GPU_TYPE:-h100}; STAGE=${STAGE:?set STAGE=A|B|C1|C2}
if [ "${SUBMIT:-0}" = 1 ]; then SB=(sbatch --parsable --account=$ACC --output=$D/logs/%x-%j.out); else SB=(sbatch --test-only --account=$ACC); fi
Q=Qwen3-14B-Base; M=Mistral-Nemo-Base-2407; G=gemma-3-12b-pt-text
sub() { "${SB[@]}" "$@" 2>&1 | tail -1; }
train() { local st=$1 data=$2 name=$3 seed=$4 dep=${5:-}; STUDENT=$st DATA=$data OUT=$AD/adapters_$st/${name}_s$seed SEED=$seed sub ${dep:+--dependency=afterok$dep} --gpus-per-node=$GPU:2 $R3/jobs/train.sbatch; }
evaluate() { local st=$1 ev=$2 adapters=$3 out=$4 dep=${5:-}; STUDENT=$st EVAL=$ev ADAPTERS="$adapters" OUT=$out sub ${dep:+--dependency=afterok$dep} --gpus-per-node=$GPU:1 $R3/jobs/eval.sbatch; }
log() { [ "${SUBMIT:-0}" = 1 ] && echo "$*" >> $D/logs/SUBMITTED.txt; echo "$*"; }
case $STAGE in
A)
  b=$(CMD="python $R3/lib/build_controls.py $RUNS $R3/inputs/floor_inputs.json" sub $R3/jobs/cpu.sbatch); log "build controls: $b"
  for st in $Q $M $G; do
    dep=""; ad=""
    for arm in translated own_unlabelled routed s_routed_$st random_mix union oracle_$st; do
      [ "$arm" = "oracle_$G" ] && continue
      i=$(train $st $RUNS/sft_$arm.jsonl $arm 2 ":$b"); log "train $st $arm s2: $i"; dep="$dep:$i"; ad="$ad ${arm}_s2=$AD/adapters_$st/${arm}_s2"
    done
    e=$(evaluate $st $RUNS/eval_floor.jsonl "$ad" $RES/floor_${st}_floor_s2.json "$dep"); log "eval $st floor s2: $e"
    e=$(evaluate $st $C24/eval_mgsm_human.jsonl "$ad" $RES/floor_${st}_mgsm_human_s2.json "$dep"); log "eval $st mgsm s2: $e"
  done ;;
B)
  for tier in high mid low; do
    dep=""; ad=""
    for arm in p000 p025 p050 p075 p100 translated; do
      i=$(train $Q $RUNS/flaw/flaw_${tier}_$arm.jsonl flaw_${tier}_$arm 1); log "train flaw $tier $arm: $i"; dep="$dep:$i"; ad="$ad flaw_${tier}_${arm}_s1=$AD/adapters_$Q/flaw_${tier}_${arm}_s1"
    done
    e=$(evaluate $Q $RUNS/flaw/eval_$tier.jsonl "$ad" $RES/flaw_${tier}_s1.json "$dep"); log "eval flaw $tier: $e"
  done ;;
C1)
  g=$(MODEL=$M NAME=mistral-core25-smoke50 LIMIT_PER_LANGUAGE=50 sub --gpus-per-node=$GPU:4 $R3/jobs/gen.sbatch); log "mistral gen smoke: $g (then read $SCRATCH/multilingual-icm/results/mistral-core25-smoke50/gen_check.txt)" ;;
C2)
  g=$(MODEL=$M NAME=mistral-core25-7473 sub --gpus-per-node=$GPU:4 $R3/jobs/gen.sbatch); log "mistral gen full: $g"
  b=$(CMD="python $R3/lib/build_gen_sets.py $SCRATCH/multilingual-icm/results/mistral-core25-7473 mistral $RUNS $R3/inputs/floor_inputs.json $C24" sub --dependency=afterok:$g $R3/jobs/cpu.sbatch); log "build mistral arms: $b"
  for st in $Q $M; do
    dep=""; ad=""
    for spec in "translated $C24/sft_translated.jsonl" "qwen_gold_filtered $C24/sft_gold_filtered.jsonl" "mistral_own_unlabelled $RUNS/sft_mistral_own_unlabelled.jsonl" "mistral_gold_filtered $RUNS/sft_mistral_gold_filtered.jsonl" "mistral_s_routed $RUNS/sft_mistral_s_routed_$st.jsonl"; do
      set -- $spec; i=$(train $st $2 $1 1 ":$b"); log "train $st $1 s1: $i"; dep="$dep:$i"; ad="$ad ${1}_s1=$AD/adapters_$st/${1}_s1"
    done
    e=$(evaluate $st $C24/eval_pipeline.jsonl "$ad" $RES/gen_${st}_pipeline_s1.json "$dep"); log "eval $st pipeline: $e"
    e=$(evaluate $st $C24/eval_mgsm_human.jsonl "$ad" $RES/gen_${st}_mgsm_human_s1.json "$dep"); log "eval $st mgsm: $e"
  done ;;
esac
[ "${SUBMIT:-0}" = 1 ] || echo "dry run only (nothing submitted). Submit with: SUBMIT=1 STAGE=$STAGE bash $0"
