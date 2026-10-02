# icm/experiments — the research record

Every experiment we ran, grouped by the question it asked. The folders are
numbered in reading order; each has a README with the answer and a table of
its files.

| folder | question | answer |
|---|---|---|
| [q01_can_voting_label_data/](q01_can_voting_label_data/) | Can languages voting on each other label the data, and does adding a model judge help? | Yes: 98% coverage at 95% on 37 × 150. A judge adds nothing. |
| [q02_can_the_model_judge_itself/](q02_can_the_model_judge_itself/) | Can the model tell its own correct chains from wrong ones? | No. Eight formulations, best contested AUC 0.674 (bar 0.70). The one at 0.918 was answer-matching in disguise. |
| [q03_does_regeneration_help/](q03_does_regeneration_help/) | Does letting a stuck language re-solve, after seeing peers, help? | Yes: 826 of 1,240 abstentions recovered, 91% correctly. |
| [q04_full_scale_headline/](q04_full_scale_headline/) | The headline run and its independent verification. | **99.23% coverage at 94.45% precision** on 55,485 pools. |
| [q05_does_search_help/](q05_does_search_help/) | Would a search over labelings (annealing, as in the ICM paper) find better labels? | No. The gold labeling scores *worse* than what voting already finds. |
| [q06_training_data_purity/](q06_training_data_purity/) | How pure is the training data this produces, and can anything clean it? | The judges cannot; world agreement looks good but is circular. |
| [q07_audits/](q07_audits/) | Is the code right? | Gold-blind claim holds. Two bugs found (fixes included); several caveats. |
| [q08_earlier_8b_study/](q08_earlier_8b_study/) | The earlier study on Qwen3-8B (chronologically first): can plain voting label the data? Does the original ICM scorer work? | Voting: ~88% coverage at ~95%. The ICM scorer: no, its score was mostly a bias for the letter "A". |
| [q09_does_the_dataset_beat_translation/](q09_does_the_dataset_beat_translation/) | Does fine-tuning on the method's chains beat fine-tuning on translated gold solutions? (pilot, 37 × 1,500) | *running* |
| [q10_pipeline_hypotheses/](q10_pipeline_hypotheses/) | The pre-registered hypotheses behind the 50-language dataset pipeline (H1–H7) | *running* |

## Conventions

- File names say what the experiment tested. `icm/provenance.json` maps each
  one to its original `~/phase0/stageNN_*.py` name, and each README lists it.
- A `.sbatch` next to a `.py` is the cluster job that ran it. GPU jobs bill
  `rrg-zhijing`, CPU jobs `def-zhijing`.
- Most scripts have `--self-test` (no data, no GPU, safe on a login node).
  Scripts without it run their whole analysis at import; run them as jobs.
- Scripts read caches from and write results to `data/ICM_work_/runs/` (see `icm/paths.py`),
  and the `.sbatch` jobs `cd` into `data/ICM_work_/runs/` first: some scripts open files
  relative to the current directory, exactly as they did from `~/phase0`. To
  run one by hand: `cd ~/multilingual_icm/data/ICM_work_/runs && PYTHONPATH=~/multilingual_icm/ICM_work_ python -m icm.experiments...`.
- The code is unchanged from the originals apart from import lines
  (`tools/check_provenance.py` verifies this). Do not edit these files to
  improve the method; edit `icm/method/` and keep these as the record.

## Historical note

The shared helpers (`norm`, `match`, `load_pools`, the tier table) were first
written inside `q02_can_the_model_judge_itself/verdict_judge_gate.py` (stage 8), and every later
experiment imports them from there. `q01_can_voting_label_data/consensus_loop_37lang.py`
(stage 17) extends the tier table from 15 to 37 languages *by patching it at
import time*. This is why the experiments import each other in odd-looking
ways. The clean versions are in `icm/method/data.py`.
