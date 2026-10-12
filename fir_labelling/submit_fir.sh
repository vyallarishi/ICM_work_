#!/usr/bin/env bash
# Fir, the 67-language labelling chain: prepare (unpack + verify) -> snapshot -> regenerate (1 GPU, ~5 h)
# -> decide labels -> build the all-67 "method" training file. Same scripts and settings as on Rorqual; only
# the account and right-sized resources are given here.
#   bash submit_fir.sh            dry run (sbatch --test-only for every job, nothing submitted)
#   SUBMIT=1 bash submit_fir.sh   submit the chain
# Options: ACCOUNT (default def-zhijing), GPU_TYPE (default h100; check `sinfo -o %G` on this cluster),
#          FROM=gen to restart at regeneration when prepare + snapshot already succeeded.
# The regeneration context window is 16,384 tokens (PREREG_4 amendment 6): some 67-language prompts exceed 4,096.
set -euo pipefail
ACC=${ACCOUNT:-def-zhijing}; GPU=${GPU_TYPE:-h100}
H=$(cd "$(dirname "$0")" && pwd); J=$HOME/multilingual_icm/data/findings/jobs
P_ARGS=(--account=$ACC $H/fir_prepare.sbatch)
S_ARGS=(--account=$ACC --job-name=label-snapshot --mem=16G --time=1:00:00 $J/label_cpu.sbatch)
G_ARGS=(--account=$ACC --gpus-per-node=$GPU:1 --cpus-per-task=16 --mem=186G --time=10:00:00 $J/label_gen.sbatch)
D_ARGS=(--account=$ACC --job-name=label-decide --mem=32G --time=2:00:00 $J/label_cpu.sbatch)
A_ARGS=(--account=$ACC --job-name=arms-all67 --mem=32G --time=2:00:00 $J/label_cpu.sbatch)
export SET=all67 LAYER=all67 ARMS=method MAX_MODEL_LEN=16384
if [ "${SUBMIT:-0}" != 1 ]; then
    for n in P S G D A; do v="${n}_ARGS[@]"; printf "%-8s " $n; CMD=snapshot sbatch --test-only "${!v}" 2>&1 | grep -o "to start.*partition [a-z_0-9]*\|rror.*"; done
    echo "dry run only; nothing submitted. Submit with: SUBMIT=1 bash $0"; exit 0
fi
if [ "${FROM:-}" = gen ]; then
    [ -d /scratch/$USER/multilingual-icm/results/labels_all67/final/pooled/generated ] || { echo "no snapshot yet: run without FROM"; exit 1; }
    G=$(sbatch --parsable "${G_ARGS[@]}");                                  echo "regenerate $G"
else
    P=$(sbatch --parsable "${P_ARGS[@]}");                                    echo "prepare    $P"
    S=$(CMD=snapshot sbatch --parsable --dependency=afterok:$P "${S_ARGS[@]}"); echo "snapshot   $S"
    G=$(sbatch --parsable --dependency=afterok:$S "${G_ARGS[@]}");             echo "regenerate $G"
fi
D=$(CMD=decide sbatch --parsable --dependency=afterok:$G "${D_ARGS[@]}");   echo "decide     $D"
A=$(CMD=arms sbatch --parsable --dependency=afterok:$D "${A_ARGS[@]}");     echo "arms       $A"
echo "logs: ~/multilingual_icm/data/findings/logs/   watch: squeue -u $USER"
