# How it works

Read [GLOSSARY.md](../GLOSSARY.md) first if a word is unfamiliar. Run
`python examples/walkthrough.py` to see every step on a toy example.

## 1. Data flow

```
generation/            GPU, vLLM
  run_experiment.py ─► generate_chains.py ─► merge_candidate_pools.py
        │
        ▼   /scratch/.../final/pooled/generated/<lang>.jsonl
        │   one row per (problem, language): question, gold_number,
        │   candidates[ {text, pred_number, finish_reason} ]
        ▼
icm/method/data.py        load_pools()          pools{ "train_<lang>_<problem>": {...} }
                          apply_diverse_views() keep ≤ 8 chains per pool
        ▼
icm/method/vote.py        initialize()          committed / contested / vetoed
                          converge()            contested pools re-vote, 4 rounds
        ▼
icm/method/regenerate.py  build_jobs()+generate()  GPU: stuck languages try again
                          weighted_decide()        CPU: re-decide with the new chains
        ▼
icm/method/run.py         decide()              → results JSON (metrics.py reads gold here)
```

## 2. The decision rule

All thresholds are 0.70.

**Load** (`data.load_pools`). Every answer is normalised so `2,190`, `2190.0`
and `2190` are the same. Chains with no parseable answer, or cut off by the
token limit, are dropped. `match(a, b)` treats answers within 0.5% as equal.

**View** (`data.apply_diverse_views`). Pools with more than 8 chains are cut to
8: one chain per distinct answer, largest groups first, then fillers.

**Initialise** (`vote.initialize`). For each pool, look at the other
languages' pluralities for the same problem:

| own share | world | state |
|---|---|---|
| ≥ 0.70 | no strong disagreement | **committed**, frozen |
| ≥ 0.70 | ≥ 70% of others hold a different answer | **vetoed**: answer None, treated as contested |
| < 0.70 | anything | **contested** |

**Converge** (`vote.update_pool`, 4 rounds). Each contested pool re-picks:

```
if ≥ 70% of the others hold an answer this language never produced → None (abstain)
else, among the answers this language did produce, take
     argmax  world_share(a) + 0.001 × local_count(a)
```

The world decides; local votes only break ties. A language never commits an
answer it did not generate itself.

**Regenerate** (`regenerate.py`). For every contested pool, show the language
2 chains supporting the world-majority answer and 1 supporting the largest
minority, and sample 4 new chains. `weighted_decide` re-runs the rule above
with the new chains counted at weight 0.5. If a new chain produces the
world's answer, the veto no longer applies and the pool is recovered. A pool
that oscillated during convergence must regenerate unanimously.

## 3. The result

`data/ICM_work_/recorded_results/stage25_results.json`, 37 languages × 1,500 problems:

| | contested answered | coverage | precision |
|---|---|---|---|
| after voting | 10,667 | 97.77% | 94.45% |
| after regeneration | 11,478 | **99.23%** | **94.45%** |

Regeneration recovered 826 of 1,240 abstained pools, 754 correctly (91.3%).
By tier: low 592/628 correct, mid 114/128, **high 48/70**. High-resource
languages that abstain are usually right and the world is wrong.

## 4. Known issues (all found by the audit in `icm/experiments/q07_audits/`)

1. **The headline used a distorted pool view.** `apply_diverse_views`
   recomputes `share` from the 8-chain view, which shrinks the top group.
   3,569 pools change landslide status. Measured effect after voting:
   precision 94.45% → 94.62%, coverage unchanged. The fixed version is
   `experiments/q07_audits/fix_pool_view_share.py`; the pipeline has not been
   re-run with it (regeneration would need to be re-generated for the new
   contested set).
2. **The veto compares answers as exact strings** (`top_world not in counts`),
   while scoring uses `match()` with a 0.5% tolerance. An answer within 0.5%
   but spelled differently still triggers the veto.
3. **Truncated chains are dropped**, and this is language-dependent: 0.1% of
   English chains vs 16.9% of Malayalam. Low-resource pools are thinner.
4. **Gold sits inside the `pools` dict** (`pools[iid]["gold"]`). The decision
   code never reads it (the gold-leak audit corrupted every gold value and
   got bit-identical decisions). Keep it that way: after changing any decision
   code, re-run `experiments/q07_audits/audit_gold_leak_1500.py`.
5. **In the experiment files**, the shared helpers live inside experiments
   (`q02_can_the_model_judge_itself/verdict_judge_gate.py` defines `norm`/`match`/`load_pools`;
   importing `q01_can_voting_label_data/consensus_loop_37lang.py` patches the language table from
   15 to 37, so import order matters), and several loop experiments copy
   `initialize`/`update_pool` instead of importing them. `icm/method/` has the
   clean versions; use those for new work.

## 5. What to improve next

In rough order of value:

1. **Run `experiments/q04_full_scale_headline/english_paraphrase_control.py`** (GPU, never
   run). It replaces 37 languages with 37 English paraphrases and decides
   whether *multilingual* is doing the work at all.
2. **Re-run the headline with the fixed pool view** (issue 1). Regenerate for
   the new contested set, then `python -m icm.method.run decide` and compare
   with `tools/compare_results.py`.
3. **Seed sweep at full scale.** `experiments/q03_does_regeneration_help/regeneration_seed_sweep.py`
   covers only the 150-problem root.
4. **Recovery for high-resource languages** (48/70). Start from
   `experiments/q03_does_regeneration_help/recovery_precision_gates.py`.
5. **Make the veto use `match()`** (issue 2) and measure the change.
6. **Truncation filter** (issue 3): keep truncated chains that have a
   parseable answer, or report per-language drop rates.

For any change to `icm/method/`, run `python -m unittest discover -s tests`
first: it will fail as soon as the method no longer matches the recorded
behaviour, which is the point. Then re-run the reproduction job.
