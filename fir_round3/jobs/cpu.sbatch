#!/usr/bin/env bash
# Generic CPU step: runs $CMD with the vLLM-environment python on PATH.   CMD="python ..." sbatch cpu.sbatch
#SBATCH --job-name=r3-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -Eeuo pipefail
module reset >/dev/null 2>&1 || true
module load python/3.11 arrow 2>/dev/null || module load python/3.11
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 XDG_CACHE_HOME=$SCRATCH/cache/xdg
source "${PROJECT:-$HOME}/venvs/multilingual-icm/bin/activate"
: "${CMD:?}"; echo "$CMD"; date; eval "$CMD"; date; echo DONE
