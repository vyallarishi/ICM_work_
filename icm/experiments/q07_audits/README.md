# audits — is the code right?

A line-by-line review of the pipeline, done as throwaway scripts. The scripts
are kept as evidence; the findings are what matter.

## Findings

| finding | scripts |
|---|---|
| **The gold-blind claim holds.** Every gold value was corrupted and the decision path re-run: output bit-identical. | `audit_gold_leak_deep`, `audit_gold_leak_1500` |
| **The headline recomputes.** Independent recount from raw pools matches `stage25_results.json`. | `audit_stage25_recompute` |
| **The judges were scored on the same chains.** Stages 8/18/19/20/41 use a byte-identical 6,149-chain sample. | `audit_repro_real`, `audit_sample_repro`, `audit_rng_order`, `audit_rng_order2` |
| **BUG: the 8-chain view distorts vote share** and depends on input order. Fix: `fix_pool_view_share.py`. Effect measured (recorded_results/RESULTS.md): precision 94.45 → 94.62 after voting. Not yet applied to the headline. | `audit_d8_real`, `audit_diverse8` |
| **BUG: arithmetic checker ignores operator precedence** (`2+3*4=14` called false); 66.7% of its flags were false alarms. Fix: `fix_arithmetic_precedence.py`. Affects the uPRM work, not the headline. | (uPRM audits, in the uPRM repository) |
| **Caveat: the veto uses exact string equality**, scoring uses `match()` (0.5%). | `audit_veto`, `audit_veto_real` |
| **Caveat: truncated chains are dropped unevenly** (0.1% English vs 16.9% Malayalam); the 15-language table silently drops languages if imported in the wrong order. | `audit_lengthbias`, `audit_loadpools`, `audit_tier` |
| The 0.001 local tie-break cannot override a world-share gap at 37 languages (it could at > 52). | `audit_eps` |
| `auc()` is correct, including ties, but hangs on NaN input. | `audit_auc`, `audit_auc2`, `audit_auc3`, `audit_auc4`, `audit_auc5`, `audit_auc_check`, `audit_auc_nan` |
| `norm()` returns None for numbers ≥ 1e28; two normalisers agree otherwise. | `audit_norm`, `audit_norm2` |
| The stage 9/17 self-tests contain an assertion that can never fail (`x or True`). | `audit_misc` |
| The +0.217 re-derivation gap (judges/truncated_peer_rederivation_at_scale) is an arm-construction artifact: the arms have different peer counts. | `audit_p3gap`, `audit_gap_s28`, `audit_gap_strat`, `audit_peercount` |
| The world-agreement harvest filter (0.926) is circular with the label. | `audit_world_circ` |
| `World` (full_scale) equals `world_context` (stage 9); `converge` equals `converge_w0`; the seed sweep covers the 150-problem root only. | `audit_world`, `audit_stage_equivalence` |
| `weighted_decide`'s unanimity branch, the generate constraint, harm accounting, iteration-order independence, metric denominators: all correct. | `audit_weighted_decide`, `audit_generate_constraint`, `audit_harm_and_order`, `audit_metrics_osc` |
| Residualisation and the hand-rolled OLS (format confound) are correct; role-swap sign convention is correct; the zero-model baseline is model-free and on the same chains. | `audit_resid`, `audit_ols`, `audit_roleswap`, `audit_peerbaseline` |
| The answer-leak detector can miss formatted numbers. | `audit_leaks`, `audit_leaks_real` |

## Data quality of each generation run

`generation_qa_phaseA.py`, `generation_qa_phaseA_30b.py`, `generation_qa_rest22.py`,
`generation_qa_smoke14b.py`, `generation_qa_smoke30b.py`: items, chains,
truncation rate, raw accuracy and landslide rate per language for each run.

## Fixes

| file | was | replaces |
|---|---|---|
| `fix_pool_view_share.py` | diverse8_fixed | `voting/consensus_loop_37lang.apply_diverse_views` |
| `fix_arithmetic_precedence.py` | arith_fixed | the uPRM `eval_equation` |

Both are new files rather than edits, so every recorded result stays reproducible.

Most audit scripts have no `--self-test`; they run their whole analysis on
import, and many need the pools on `/scratch`. Run them as jobs.
