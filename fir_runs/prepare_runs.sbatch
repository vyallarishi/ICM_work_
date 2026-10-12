#!/usr/bin/env bash
# Fir, CPU: verify the downloaded Mistral and Gemma against Rorqual's checksums, build the text-only Gemma student
# exactly as on Rorqual (q09 extract_text_model.py, training env), and verify it too. Any mismatch stops the chain.
#SBATCH --job-name=fir-prepare-runs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/findings/logs/%x-%j.out
set -Eeuo pipefail
module load python/3.11
M=/scratch/$USER/multilingual-icm/assets/models; T=/scratch/$USER/fir_runs
( cd $M/Mistral-Nemo-Base-2407 && sha256sum -c --quiet $T/model_Mistral-Nemo-Base-2407.sha256 ) && echo "Mistral: identical to Rorqual's"
( cd $M/gemma-3-12b-pt && sha256sum -c --quiet $T/model_gemma-3-12b-pt.sha256 ) && echo "gemma-3-12b-pt: identical to Rorqual's"
source ${PROJECT:-$HOME}/venvs/sm-icm/bin/activate
export PYTHONPATH=$HOME/multilingual_icm/ICM_work_ TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
[ -f $M/gemma-3-12b-pt-text/config.json ] || python -u -m icm.experiments.q09_does_the_dataset_beat_translation.extract_text_model \
    --src $M/gemma-3-12b-pt --dst $M/gemma-3-12b-pt-text
( cd $M/gemma-3-12b-pt-text && sha256sum -c --quiet $T/model_gemma-3-12b-pt-text.sha256 ) && echo "gemma-3-12b-pt-text: identical to Rorqual's"
date; echo DONE
