#!/usr/bin/env bash
# The offline phase of setup/build_data.py as a CPU job (~1 h; too long for a login node).
# Run the download phase on a login node first. Submit with:
#     source ./config.env && sbatch --account=$ACCOUNT_CPU setup/build_data.sbatch
#SBATCH --job-name=build-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=build-data-%j.out
set -Eeuo pipefail
: "${BUNDLE:?source ./config.env before sbatch}"
if command -v module >/dev/null 2>&1 && [ -n "$MODULES_VLLM" ]; then module reset >/dev/null 2>&1 || true; module load $MODULES_VLLM; fi
source "$VENV_VLLM/bin/activate"
cd "$BUNDLE"
python -u setup/build_data.py build
