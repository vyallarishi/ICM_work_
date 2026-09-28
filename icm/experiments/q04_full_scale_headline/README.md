# full_scale — the headline run and its verification

**37 languages × 1,500 problems = 55,485 pools: 99.23% coverage at 94.45%
precision, gold-blind.** Data: `qwen3-14b-full37-1500`.

| file | was | what it does |
|---|---|---|
| `full_scale_37x1500.py` | stage25_scale | **the headline run.** Voting (from `q01_can_voting_label_data/consensus_loop_15lang.py`) + one regeneration round (from `q03_does_regeneration_help/regeneration_15lang.py`). Adds a problem-indexed `World` so the loop is not quadratic at 55k pools; its `--self-test` proves that gives identical results on the 150-problem root. `--phase gen` (GPU) writes `data/ICM_work_/runs/stage25_regen.jsonl`; `--phase decide` (CPU) writes `stage25_results.json` |
| `verify_full_scale_independently.py` | verify_stage25 | recomputes coverage, precision and recovery from the raw files with its own loops. Prints MISMATCH on any difference; all numbers matched |
| `english_paraphrase_control.py` | stage31_paraphrase_control | **never run.** Replaces 37 languages with 37 English paraphrases of each problem. If it matches the cross-lingual numbers, "multilingual" is not doing the work. Built and self-tested; needs a GPU |
| `reproduce_headline.sbatch` | (new) | CPU job: runs `icm/method`, the original code, and the independent verification, and diffs each against `data/ICM_work_/recorded_results/stage25_results.json` |
| `full_scale_37x1500.sbatch`, `english_paraphrase_control.sbatch` | | the original GPU jobs |

The clean implementation of what this folder runs is `icm/method/`.
