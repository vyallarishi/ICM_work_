# judges — can the model tell its own correct chains from wrong ones?

**Answer: no.** The bar was AUC ≥ 0.70 on contested pools (where the vote is
split, so a judge would actually matter). Eight formulations were tried on
the same 6,149-chain sample; the best honest one reached 0.674. The one that
reached 0.918 was beaten by a baseline with no model at all.

| file | was | idea | contested AUC |
|---|---|---|---|
| `verdict_judge_gate.py` | stage8_judge_gate | "Is this chain correct? Yes/No", with 3 in-language demos. **Also defines the shared helpers `norm`, `match`, `auc`, `load_pools` that every later file imports.** | 0.636 |
| `likelihood_ratio_judge.py` | stage18_lr_judge | log P(chain \| agreeing demos) − log P(chain \| disagreeing demos), over all tokens | 0.536 |
| `likelihood_ratio_autopsy.py` | lr_autopsy | why it failed: chain length dominates | diagnostic |
| `peer_conditioned_judge.py` | stage19_peer_judge | show the judge 3 chains for the same problem from other languages | 0.918 |
| `peer_judge_zero_model_baseline.py` | peer_baseline | **the kill**: just count how many peers' answers match the chain's answer, no model | 0.946 |
| `continuation_consistency_judge.py` | stage20_continuation | cut the chain at 60%, sample 4 continuations, count how many reach its own answer | 0.649 |
| `ensemble_judge_in_loop.py` | stage21_ens_loop | verdict + continuation, z-scored and summed, inside the voting loop | 0.674; no gain in the loop |
| `judge_channel_correlation.py` | channel_ensemble | are the ~0.65 channels the same signal? (Spearman, ensemble AUC) | diagnostic |
| `generative_mutual_predictability.py` | stage26_sa_score | S(a) = P(language re-derives a \| 3 peers that derived a), by sampling | selectivity 52% (chance) |
| `truncated_peer_rederivation.py` | stage27_truncated | same, but peers' last step removed so the model must finish the derivation | gap +0.131 |
| `truncated_peer_rederivation_at_scale.py` | stage28_trunc_scale | powered to 11,907 pools with harder truncation | gap +0.217, but see audits: an artifact of peer count; matched it is +0.032 |
| `marker_token_comparison.py` | stage41_markers | the same judge with `Yes/No`, `+/-`, `True/False`, `A/B` markers, everything else fixed | 0.636 / 0.649 / 0.634 / 0.629: the marker is not what is missing |
| `format_confound.py` | format_confound | how much of the judge margin is formatting (LaTeX, bold, length)? R² = 0.025; centring per language helps more | diagnostic |
| `format_confound_ablation.py` | format_confound_ablate | language offsets vs format features | diagnostic |
| `rebuild_gate_sample.py` | quality_sample | rebuilds stage 8's sample to re-join margins to chains | helper |

The `.sbatch` files next to the scripts are the GPU jobs that ran them.

Why it matters: the original ICM method rests on the model judging its own
labels. At 14B, on maths, it cannot do that well enough on the pools where it
would matter. Cross-lingual agreement is the signal that works.
