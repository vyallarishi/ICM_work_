# qwen3_8b_study — the earlier study on Qwen3-8B

Before the 14B work, the same questions were asked of Qwen3-8B-Base on
37 languages × 200 problems (data under `/scratch/rvyalla/sm-icm/`). Two
things came out of it that shaped everything after:

1. **Plain voting with a cross-lingual check already works**: about 88%
   coverage at 95% precision (`rung7_small_pools_and_routing.py`).
2. **The original ICM scorer does not**: its score was dominated by a
   preference for the marker letter "A", about 5× the content signal
   (`icm_scorer_autopsy.py`, `icm_scorer_probe_gpu.py`).

The "rungs" add one part at a time to a minimal baseline.

| file | was | what it tested |
|---|---|---|
| `phase0_diagnostics.py` | phase0 | was gold ever in the pool? do languages make the *same* wrong answer? is confidence calibrated? |
| `rung0_1_vote_baseline.py` | rung01_vote_baseline | vote, commit at ≥ 70%, else abstain; choose the rule by measurement |
| `rung2_execute_arithmetic.py` | rung2_execution | have the model transcribe each chain's arithmetic to Python and execute it |
| `rung3_reasoning_structure.py` | rung3_equations | compare chains by the equations they compute (`<<a op b=c>>`), not their answer. **Defines the equation parser used later** |
| `rung4_crosslingual_structure.py` | rung4_crosslingual_structure | break ties by matching a chain's structure to other languages' |
| `rung5_structure_coherence.py` | rung5_anna_karenina | correct derivations are near-identical across languages; wrong ones vary. Score by coherence of support |
| `rung6_world_check_and_jury_size.py` | rung6_worldcheck_ensemble | flag landslides the world disagrees with; how many reference languages are needed |
| `rung7_small_pools_and_routing.py` | rung7_smalln_routing | the pipeline with only 4–8 chains per pool; route abstentions to a judge |
| `icm_scorer_autopsy.py` | e0_scorer_autopsy | why the ICM scorer failed (marker bias) |
| `icm_scorer_probe_gpu.py` | e3_scorer_probe | rescoring under fp32 / Yes-No markers / no demos |
| `stage5_judge_on_residual.py` | stage5_judge_probe | the best judge on the pools the pipeline abstains on |
| `stage6_user_spec_config.py` | stage6_userspec | a user-specified configuration (4 chains, global-first, 3-demo judge) head to head |
| `stage6_diagnosis.py`, `stage6_fixes.py`, `stage6_escalation_ladder.py` | stage6_diag, stage6_fix, stage6_ladder | where its errors came from; escalate 4 → 8 → 16 chains |
| `stage7_diverse4_with_judge.py` | stage7_diverse4_mp | 4-chain diverse views with a judge on the residual |
| `diverse4_pipeline.py`, `gold_coverage_in_4_chains.py`, `gold_coverage_in_diverse4.py` | | small checks on views and gold coverage |
| `m2_tiebreak_audit.py` | m2_audit | independent re-derivation of the cross-lingual tie-break |

Results: `recorded_results/rung*_results.json`, `stage5–7_results.json`,
`e0_results.json`, `e3_results.json`, `phase0_results.json`.
