# q10: the hypotheses behind the dataset pipeline, pre-registered

Each hypothesis is a claim the paper needs. The pass/fail bar is written here
and in the script's docstring **before** running, and every script prints
`HOLDS` / `FAILS` against it. Data: 37 languages × 1,500 problems
(`qwen3-14b-full37-1500`). Label quality is always scored by exact numeric
equality with gold; gold never enters a decision.

**Gold budget.** The 1,500 problems are split once (seeded hash) into **300
calibration problems** and **1,200 labelling problems**. Anything fit with gold
(the probe, the confidence model, the fusion weights) sees only calibration
problems of *other* languages and is scored only on labelling problems of the
held-out language. The paper's claim is therefore "gold for 300 problems in
some languages labels everything else", stated as such.

| | hypothesis | test | bar |
|---|---|---|---|
| **H1** | the audited rule fixes (v2) cost no label quality | `h1_rules.py`: voting only, record vs v2 rules | precision ≥ record − 0.2, coverage ≥ record − 0.5 |
| **H2** | a label-free confidence, fit on other languages' calibration problems, is useful on unseen languages and problems | `h2_confidence.py` | (a) AUC ≥ 0.90; (b) the threshold keeping 90% of training pools keeps ≥ 85% of held-out pools at ≥ 97.5% precision; (c) on uncertain pools, ECE ≤ 0.05 with n ≥ 500; (d) ≥ 0.97 precision at confidence ≥ 0.95 in every tier, n ≥ 200 each |
| **H3a** | fusing the probe with the vote (weight a function of confidence) fixes more labels than it breaks in high-resource languages | probe_work_ `q10_fusing_vote_and_probe/fusion.py`, disjoint by language and problem | fixed > broke, sign test p < 0.01 |
| **H3b** | the probe's value is tier-dependent: it helps high-resource, not low-resource | same, per tier | net gain per pool high > low, and low not significant (p ≥ 0.05) |
| **H4a** | fine-tuning on the method's data beats translated solutions | `q09_…`: 4 arms, same pairs, 2 seeds, paired bootstrap on 9,250 test questions | method − translated ≥ +1.0 point, CI > 0 |
| **H4b** | the gain comes from the labelling, not just from training on the model's own text | same; control `own_unlabelled` (own chains, no labelling) | method − own_unlabelled, CI > 0 |
| **H4c** | the gain is not just more tokens | same; control `method_lenmatch` | method_lenmatch − translated, CI > 0 |
| **H5a** | correct labels per pool grow with jury size and saturate | `h5_scaling.py` (a) | +≥ 2 points from 1 to 16 languages; < 0.5 from 24 to 36 |
| **H5b** | ~8 samples per language suffice | `h5_scaling.py` (b) | coverage rises with samples; precision at 8 within 1 point of all |
| **H6** | translated solutions carry artefacts native chains do not | `h6_translationese.py`: third-script letters, same pairs | translated worse in ≥ 2/3 of non-Latin languages, pooled p < 0.01 |
| **H7a** | the pipeline audits itself on new problems: precision on number-perturbed variants (derived answers, no human labels) matches precision on their originals | `h7_*`: 500 variants × 37 languages | gap ≤ 3 points (answer key checked: suspect variants flagged, 30 written out for a hand check) |
| **H7b** | variants are solved, not remembered | `h7_audit.py` | ≤ 2% of commits equal the original problem's answer |

Already tested and **rejected** (kept in the record): the probe ranks sound
reasoning above lucky guesses with the same answer (probe_work_ q08, every
pooling at chance); English paraphrases label as well as other languages
(q04 paraphrase control: H0 held). Regeneration adds no label information
beyond the vote (overwrite control 94.9% vs 91.3%); it stays in the data
pipeline as guided re-sampling, the only way an abstained pool yields a chain.

| file | what |
|---|---|
| `common.py` | exact scoring, tiers, folds, AUC |
| `h1_rules.py`, `h2_confidence.py`, `h5_scaling.py` → `cpu_hypotheses.sbatch` | CPU |
| `h6_translationese.py` → `h6.sbatch` | CPU, after the q09 training sets exist |
| `h7_build_variants.py` → `h7_build_variants.sbatch` | CPU: variants dataset on /scratch |
| `../../../generation/cluster/h7_generate_variants.sbatch` | GPU 4 × H100: chains for the variants, record settings |
| `h7_audit.py` → `h7_audit.sbatch` | CPU: label and score the variants |

Results: `data/ICM_work_/runs/h*.json`; logs in `data/ICM_work_/logs/`.
