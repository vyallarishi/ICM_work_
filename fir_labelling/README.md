# 67-language labelling on Fir

Runs the project's labelling method (unchanged) on all 67 languages of the Qwen3-14B-Base record, and builds the all-67 `method` training file. The scripts in `jobs/` are byte-identical copies of the Rorqual ones (`data/findings/jobs/`). They use absolute paths under `/home/$USER/multilingual_icm` and `/scratch/$USER/multilingual-icm`. Your username is the same on both clusters, so `setup_fir.sh` recreates that layout and nothing needs editing.

## Steps

1. **Data:** nothing to copy. The compressed Qwen record ships with this branch in `transfer/` (7 parts under 100 MB, plus checksums), and `setup_fir.sh` rebuilds and verifies it.
2. **Clone** this branch to the exact path:
   ```
   git clone -b fir-labelling https://github.com/vyallarishi/ICM_work_.git ~/multilingual_icm/ICM_work_
   ```
3. **Set up** on a login node: `bash ~/multilingual_icm/ICM_work_/fir_labelling/setup_fir.sh`. It builds the environment from PyPI with `uv` (not the cluster wheelhouse) and downloads Qwen3-14B-Base.
4. **Dry run:** `bash ~/multilingual_icm/ICM_work_/fir_labelling/submit_fir.sh` runs `sbatch --test-only` for every job and submits nothing.
5. **Submit:** `SUBMIT=1 bash ~/multilingual_icm/ICM_work_/fir_labelling/submit_fir.sh`

## The chain

| Step | Job | Resources | Expected time |
|---|---|---|---|
| 1 | prepare: unpack the record, verify all 67 files and the model against Rorqual's checksums | CPU | ~10 min |
| 2 | snapshot | CPU | ~3 min |
| 3 | regenerate stuck problems | 1 GPU, 186 GB | ~5 h (the 25-language run took 1 h 42) |
| 4 | decide labels | CPU | ~5 min |
| 5 | build the all-67 method training file | CPU | ~3 min |

**Check:** every vLLM log must contain `Using FlashAttention version 3` and `Using FlashInfer for top-p & top-k sampling`. If not, the environment is a slow fallback; stop and rebuild it.

## Results to send back to Rorqual

| File | Size | What |
|---|---|---|
| `/scratch/$USER/multilingual-icm/results/labels_all67/labels.json` | ~15 MB | the labels |
| `/scratch/$USER/multilingual-icm/results/labels_all67/regen.jsonl` | ~10 MB | regenerated answers |
| `~/multilingual_icm/data/findings/results/labels_all67_summary.json`, `arms_all67_coverage.json` | small | coverage and precision |
| `~/multilingual_icm/data/findings/runs_all67/sft_method.jsonl` | ~450 MB | the all-67 method training file (also goes to Yara) |
| `~/multilingual_icm/data/findings/logs/` (the five jobs' logs) | small | |
