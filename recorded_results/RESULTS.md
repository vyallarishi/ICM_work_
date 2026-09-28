# Recorded results

Every JSON in `data/ICM_work_/recorded_results/` (kept outside git) is the output of one run, exactly as produced.
The file names keep the original stage numbers; the script that wrote each one
states its pre-registered prediction in its docstring.

## The headline

`stage25_results.json`, from `icm/experiments/q04_full_scale_headline/full_scale_37x1500.py`:

| | coverage | precision |
|---|---|---|
| after voting | 97.77% | 94.45% |
| after regeneration | **99.23%** | **94.45%** |

Recovered 826 of 1,240 abstained pools, 754 correct.

## Effect of the pool-view fix (measured, not yet applied)

From `~/phase0/fix_impact.log`, on the same 55,485 pools, after voting:

| | old view (headline) | fixed view |
|---|---|---|
| landslide pools | 44,181 | 46,880 |
| contested pools | 11,907 | 9,336 |
| coverage | 97.77% | 97.76% |
| precision | 94.45% | 94.62% |

3,569 pools change status and 2,661 change plurality, but the world vote
absorbs almost all of it.

## All files

| file | script | what it measured |
|---|---|---|
| `stage25_results.json` | full_scale/full_scale_37x1500 | **the headline** |
| `stage24_results.json` | regeneration/regeneration_seed_sweep | regeneration at 5 seeds (150-problem root) |
| `stage23_results.json` | regeneration/regeneration_37lang | regeneration, 37 languages × 150 |
| `stage22_results.json` | regeneration/regeneration_15lang | regeneration, 15 languages × 150 (51/79 recovered at 98%) |
| `stage17_results.json` | voting/consensus_loop_37lang | voting loop at 37 languages |
| `stage9_results.json`, `stage9d_results.json` | voting/consensus_loop_15lang, _diverse4_views | voting loop, 15 languages, with judge arms |
| `stage8_results.json` | judges/verdict_judge_gate | verdict judge: contested AUC 0.636 |
| `stage18_results.json` | judges/likelihood_ratio_judge | 0.536 |
| `stage19_results.json` | judges/peer_conditioned_judge | 0.918 (killed by the zero-model baseline, 0.946) |
| `stage20_results.json` | judges/continuation_consistency_judge | 0.649 |
| `stage21_results.json` | judges/ensemble_judge_in_loop | 0.674; no in-loop gain |
| `stage26_results.json` | judges/generative_mutual_predictability | selectivity at chance |
| `stage27_results.json` | judges/truncated_peer_rederivation | gap +0.131 |
| `stage28_results.json` | judges/truncated_peer_rederivation_at_scale | gap +0.217, shown to be an artifact |
| `stage41_results.json` | judges/marker_token_comparison | marker tokens: best 0.649 (`+/-`), bar not passed |
| `stage29_results.json` | search/static_overlap_energy | energy landscape: dead |
| `stage30_results_trunc2*.json` | search/rederivation_energy | re-derivation energy: dead |
| `stage10_*`, `stage13_*`, `stage15_*`, `stage16_*` | voting/judge_weight_* , judge_first_init_*, judge_loop_37lang | superseded judge-in-loop runs |
| `stage12_cpu_results.json` | voting/diverse8_vs_diverse4_cpu | choosing the 8-chain view |
| `stage5_*`, `stage6_*`, `stage7_*`, `rung*_results.json`, `e0_*`, `e3_*`, `phase0_*` | qwen3_8b_study/ | the earlier Qwen3-8B study |

To check a number, re-run the script (see its `.sbatch` next to it) and diff:

```bash
python tools/compare_results.py runs/<file>.json data/ICM_work_/recorded_results/<file>.json
```
