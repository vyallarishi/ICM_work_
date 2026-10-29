# Round 3 on Fir: controls, S-rule, third student, third generator, flaw tolerance

Pre-registration: `PREREG_round3.md` (read first). Everything reuses the core-24 recipe and the repo's scripts
unchanged; only data and job paths differ. Prior results and the analysis this follows are on Rorqual under
`data/native_vs_sft/`.

## Prerequisites on Fir (from the earlier wave)
`~/multilingual_icm/ICM_work_` on branch `fir-labelling`, both venvs (`fir_runs/setup_runs.sh`), the three
models under `$SCRATCH/multilingual-icm/assets/models/`, and `runs_core24/` files under
`~/multilingual_icm/data/findings/`. `setup.sh` checks all of this.

## Steps
```
# on Fir, login node
cd ~/multilingual_icm/ICM_work_ && git pull                      # brings fir_round3/
scp rvyalla@rorqual.alliancecan.ca:/scratch/rvyalla/fir_round3/round3.tar.zst $SCRATCH/   # ~0.5 GB
bash fir_round3/setup.sh                                          # verify, unpack, place the dataset
STAGE=A  bash fir_round3/submit.sh                                # dry run; then:
SUBMIT=1 STAGE=A  bash fir_round3/submit.sh                       # Tier 1: controls build -> 19 trainings -> 6 evals
SUBMIT=1 STAGE=B  bash fir_round3/submit.sh                       # Tier 2b: 18 short trainings -> 3 evals
SUBMIT=1 STAGE=C1 bash fir_round3/submit.sh                       # Tier 2a smoke: Mistral generation, 50 problems
cat $SCRATCH/multilingual-icm/results/mistral-core25-smoke50/gen_check.txt   # must say GATES: PASS
SUBMIT=1 STAGE=C2 bash fir_round3/submit.sh                       # full generation -> arms -> 10 trainings -> 4 evals
```
Stages are independent of each other and can be submitted in any order once setup is done (C2 after C1 passes).
`squeue -u $USER` to watch; job IDs are appended to `~/multilingual_icm/data/native_vs_sft/fir_round3/logs/SUBMITTED.txt`.

## What comes back (small files)
`~/multilingual_icm/data/native_vs_sft/fir_round3/results/*.json` (every evaluation; per-item correctness
included), `runs/controls_counts.json`, `runs/mistral_agreement.json`, `runs/mistral_coverage.json`, the
`gen_check.txt`, and the job logs. Copy that folder back to Rorqual; the scorers and bootstrap CIs run there.

## If a resource flag does not fit Fir's nodes
`ACCOUNT`, `GPU_TYPE` are variables in `submit.sh`; CPU/memory numbers are in the `#SBATCH` lines of
`jobs/*.sbatch`. None of these change results.
