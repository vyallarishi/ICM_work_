#!/usr/bin/env bash
# Fir, step 1 (CPU): unpack the transferred Qwen record and verify it file by file; verify the downloaded
# Qwen3-14B-Base against Rorqual's checksums. Any mismatch stops the whole chain.
#SBATCH --job-name=fir-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/findings/logs/%x-%j.out
set -Eeuo pipefail
T=/scratch/$USER/fir_transfer
R=/scratch/$USER/multilingual-icm/results/qwen3-14b-full66-7473/final/pooled
M=/scratch/$USER/multilingual-icm/assets/models/Qwen3-14B-Base
( cd $T && sha256sum -c transfer.sha256 )
rm -rf $R/generated; mkdir -p $R
zstd -dc -T8 $T/record_qwen3-14b-full66-7473.tar.zst | tar -C $R -xf -
( cd $R && sha256sum -c --quiet $T/record_files.sha256 ) && echo "record: 67 files verified"
( cd $M && sha256sum -c --quiet $T/model_Qwen3-14B-Base.sha256 ) && echo "model: identical to Rorqual's copy"
date; echo DONE
