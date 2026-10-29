#!/usr/bin/env bash
# 2-shot generation for a new generator on the core 25 languages: the record's manifest (best_regimes_v2:
# 8/8/4 chains at T 0.2/0.5/0.8, seeds 501-503, 768 tokens, 2 translated demonstrations), one vLLM replica per
# GPU, languages sharded across replicas, then the repo's merge. Generation repo scripts unchanged.
#     MODEL=Mistral-Nemo-Base-2407 NAME=mistral-core25-7473 [LIMIT_PER_LANGUAGE=50] sbatch gen.sbatch
#SBATCH --job-name=r3-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -Eeuo pipefail
module reset >/dev/null 2>&1 || true
module load python/3.11 cuda
: "${MODEL:?}"; : "${NAME:?}"; : "${SLURM_TMPDIR:?}"; : "${CUDA_VISIBLE_DEVICES:?}"
REPO=$HOME/multilingual_icm/ICM_work_/generation; ASSETS=$SCRATCH/multilingual-icm/assets; RESULT_ROOT=$SCRATCH/multilingual-icm/results/$NAME
MANIFEST=$REPO/configs/best_regimes_v2.json
LANGUAGES=$(python3 -c "import json; print(' '.join(json.load(open('$HOME/multilingual_icm/ICM_work_/fir_round3/inputs/floor_inputs.json'))['core25']))")
export HF_HOME=$SCRATCH/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=true VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=$SCRATCH/cache/vllm TRITON_CACHE_DIR=$SCRATCH/cache/triton XDG_CACHE_HOME=$SCRATCH/cache/xdg
source "${PROJECT:-$HOME}/venvs/multilingual-icm/bin/activate"; cd "$REPO"
echo "model=$MODEL name=$NAME limit=${LIMIT_PER_LANGUAGE:-all} languages=$LANGUAGES"; echo "manifest: $(sha256sum "$MANIFEST")"
mkdir -p "$SLURM_TMPDIR/model" "$SLURM_TMPDIR/data" "$RESULT_ROOT/logs"
rsync -a "$ASSETS/models/$MODEL/" "$SLURM_TMPDIR/model/"; rsync -a "$ASSETS/datasets/gsm8k-106/" "$SLURM_TMPDIR/data/"; date
python scripts/preflight_offline.py --model-dir "$SLURM_TMPDIR/model" --dataset-dir "$SLURM_TMPDIR/data" --language af
IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"; N=${#GPU_IDS[@]}; SHARDS=$RESULT_ROOT/workers_${N}gpu; mkdir -p "$SHARDS"; pids=()
for ((r = 0; r < N; r++)); do
  cmd=(python scripts/run_experiment.py --manifest "$MANIFEST" --model "$SLURM_TMPDIR/model" --dataset "$SLURM_TMPDIR/data" --output-dir "$SHARDS/shard_$r"
       --language-shard-count "$N" --language-shard-index "$r" --generate-only --languages $LANGUAGES)
  [[ -n "${LIMIT_PER_LANGUAGE:-}" ]] && cmd+=(--limit-per-language "$LIMIT_PER_LANGUAGE")
  mkdir -p "$SHARDS/shard_$r"; CUDA_VISIBLE_DEVICES="${GPU_IDS[$r]}" OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / N)) "${cmd[@]}" > "$RESULT_ROOT/logs/shard-$r-$SLURM_JOB_ID.log" 2>&1 & pids+=("$!")
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done; [[ $status -eq 0 ]] || { echo "a replica failed; see $RESULT_ROOT/logs/" >&2; exit 1; }
python scripts/merge_experiment_shards.py --shards-root "$SHARDS" --output-dir "$RESULT_ROOT/final" --overwrite
python "$HOME/multilingual_icm/ICM_work_/fir_round3/lib/gen_check.py" "$RESULT_ROOT" | tee "$RESULT_ROOT/gen_check.txt"
date; echo "DONE $RESULT_ROOT/final"
