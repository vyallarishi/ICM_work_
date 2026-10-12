#!/usr/bin/env bash
# Labelling, CPU steps (label_run.py snapshot | decide) or arm building (build_arms.py). -> def-zhijing
#     CMD=snapshot SET=core25 sbatch label_cpu.sbatch
#     CMD=decide   SET=core25 sbatch label_cpu.sbatch
#     CMD=arms LAYER=core24 ARMS="own_unlabelled gold_filtered" sbatch label_cpu.sbatch
#SBATCH --account=def-zhijing
#SBATCH --job-name=label-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=186G
#SBATCH --time=24:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/findings/logs/%x-%j.out
set +e
module reset >/dev/null 2>&1
module load python/3.11
set -Eeuo pipefail
: "${CMD:?set CMD=snapshot|decide|arms}"
PROJECT_STORAGE="${PROJECT:-${HOME}}"
J=/home/rvyalla/multilingual_icm/data/findings/jobs
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
echo "pre-registration on record: $(sha256sum "$J/PREREG_4_languages.md")"; date
if [ "$CMD" = arms ]; then
    : "${LAYER:?}"; : "${ARMS:?}"
    python -u "$J/build_arms.py" --layer "$LAYER" --arms $ARMS
else
    : "${SET:?}"
    python -u "$J/label_run.py" "$CMD" --set "$SET"
fi
date; echo DONE
