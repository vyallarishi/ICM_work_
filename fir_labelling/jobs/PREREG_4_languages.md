# Pre-registration 4: language sets for the shared-errors experiments

Frozen 2026-10-09, before any generation in the new languages. Record this file's sha256 in every job log.

## Sources (external, dated; no result from our own experiments was used)
- Translation-quality table "Multilingual Math QE - Full Scores": CometKiwi, MetricX and GEMMA scores for 105 languages. Stored in `data/findings/inputs/`.
- Common Crawl language shares, crawl CC-MAIN-2026-39 (commoncrawl.github.io/cc-crawl-statistics/plots/languages).
- Selection code: `jobs/language_selection.py`. Output: `results/language_selection.json` (job 22806893). Input sha256s are printed in `logs/findings-langsel-22806893.out`.

## Rules
| Rule | Definition |
|---|---|
| Quality gate | All three QE scores present; CometKiwi ≥ 0.82; MetricX ≤ 5.5; GEMMA ≥ 90; living natural language with its own Common Crawl share |
| Tiers | Share of Common Crawl pages: high ≥ 1%, mid 0.1–1%, low < 0.1% |
| Per tier (core only) | 8 languages: 5 Latin-script + 3 other-script, from ≥ 2 non-Latin script families; every MGSM language in that tier included |
| Whole set (core only) | ≤ 1 language per subgroup; ≤ 2 per WALS genus |
| Choice (core only) | Minimise the spread of tier-mean CometKiwi, MetricX and GEMMA, each scaled by its variance across eligible languages. Ties go to more genera, then alphabetical order. Search is exhaustive over all 10,530 valid sets. |

## Layer 1: core set (24 languages + English), used in every experiment
| Tier | Languages |
|---|---|
| High | de, es, fr, tr, id, ja, ru, zh-cn |
| Mid | vi, hu, da, sk, lt, th, bn, ko |
| Low | sw, sq, eu, uz, tl, te, hy, kn |

Tier means (CometKiwi / MetricX / GEMMA):
- high: 0.859 / 3.30 / 95.68
- mid: 0.858 / 3.85 / 95.65
- low: 0.853 / 4.40 / 94.55

Imbalance score is 0.528. Over 100 uniformly random valid sets the score is median 0.960, range 0.582–1.596.

English is the source language and reference; it is not in any tier.

## Layer 2: extension set (all 66 languages passing the gate; includes the core)
Used only for the pool-level pipeline: generation, voting, shared/local classification.

| Tier | Count | Languages |
|---|---|---|
| High | 13 | ru de ja es fr zh-cn pt it pl nl tr cs id |
| Mid | 22 | vi ko uk sv fa ar ro hu da el sk th no bg sr he hi hr ca lt sl bn |
| Low | 31 | ms az bs ne sq ka ta kk mk eu gl hy uz ur ml te be mn kn tl gu cy sw km ky pa af or lo am yi |

## Analysis commitments
1. **Primary claims are per-language and continuous; no tier cut enters any test.**
   - Shared-error count vs log₁₀ Common Crawl share: equivalence test that |slope| is below a bound fixed before analysis (to be set in the analysis prereg).
   - Local-error count vs log₁₀ Common Crawl share: one-sided test.
   - Fit on the extension set as a mixed model with a random intercept for genus, with the three QE scores as covariates.
   - Repeat on the core set.
2. **Tiers are used only for figures and per-tier summaries**, and only on the core set.
3. **Robustness:**
   - every headline result is repeated on the 100 random valid core sets and the top-50 sets in `language_selection.json`;
   - the share of sets in which it holds is reported.
4. **Human-translation control:** the 10 MGSM languages (de es fr ja ru zh-cn | th bn | sw te) are rerun on MGSM human translations of GSM8K test problems 0–249, and the local/shared split is compared with the pipeline translations.
5. **Sensitivity:**
   - quality gate loosened (CometKiwi ≥ 0.80, MetricX ≤ 6.0) and tightened (≥ 0.84, ≤ 5.0);
   - leave-one-language-out;
   - leave-one-genus-out.

## Known limitations (stated in the paper)
- Common Crawl share stands in for the generator's undisclosed pretraining mix.
- CLD2 language detection merges much Malay into Indonesian.
- The low tier's mean MetricX (4.40) and GEMMA (94.55) are worse than the high tier's (3.30 / 95.68). No valid set closes this gap, so translation quality is a covariate (commitment 1) and is tested directly (commitment 4).

## Amendment 1 (2026-10-09, before any run): data scale, and the experiments that justify 24 languages
**Data.** The core 24 + en use the FULL GSM8K splits: train 7,473 problems, test 1,319 problems. The 66 extension languages also use the full splits (7,473 / 1,319), so both layers share every problem.

**Why 24 is enough: tested on the 66, each with a pass/fail bar fixed here.**

| # | Experiment (66 languages) | Holds if |
|---|---|---|
| E1 | Saturation: label precision, coverage and shared-error share vs jury size k = 2…66 (100 random juries per k) | the curve has plateaued by k = 24: the k = 66 value lies within the k = 24 jury-to-jury 95% range |
| E2 | Labels from the 24-language vote vs the 66-language vote, on the same 7,473 problems × 24 languages | ≥ 98% identical committed answers, and the precision difference has a 95% CI inside ±1 point |
| E3 | Representativeness: every headline statistic on the core 24 vs 1,000 random 24-subsets of the 66 (any composition) | the core value lies in the central 90% of that distribution |
| E4 | Main claims (shared flat, local falls with resource) on all 66, continuous, genus random effect, QE covariates | same sign as on the core and CI excludes 0 (local); equivalence bound met (shared) |
| E5 | Transfer: core-trained adapters evaluated zero-shot on the 42 held-out languages (eval only) | method − translated has the same sign on held-out as on core languages |
| E6 | Specification curve: every 2–5-tier contiguous cut of the 66 | the claim's sign is the same in ≥ 95% of splits |
| E7 | Selection-rule ablations: rerun the selection under each rule removed or changed (gate loose/strict; no MGSM forcing; no script rule; ≤ 2 per subgroup) and recompute the headline statistics on each resulting set | the claims hold on every alternative set |
| E8 | Leave-one-language-out and leave-one-genus-out on the 66 | no single language or genus flips a claim |
| E9 | Translation quality: partial correlations on the 66; MGSM human vs pipeline on the 10 MGSM languages | the local-error/resource relation survives controlling for QE |

## Amendment 2 (2026-10-09, before any run): generators
- **Primary generator:** Qwen3-14B-Base, run on all 66 languages + en.
- **One extra generator for now:** gemma-3-12b-pt, a different family at matched size and the most multilingual base model we hold. It runs on the core 24 + en, full data, and must pass a smoke test first (no empty outputs, no template echoes, no memorised GSM8K solutions). If it fails, the fallback is Mistral-Nemo-Base-2407.
- **Second-view comparison** (same model in another language vs another model in the same language): accuracy-adjusted error overlap (kappa / CAPA), with equal numbers of views on each side.
- **Deferred, and stated as a limitation until run:** a third family and the scale pair (Qwen3-8B-Base).
- **Fine-tuning students are unchanged:** Qwen3-14B-Base, Mistral-Nemo-Base-2407, gemma-3-12b-pt-text, with the same recipe, seeds and evaluation sets for every arm, so the gold and method arms are directly comparable.

## Amendment 3 (2026-10-10, before the run): all-67 fine-tuning ablation
- **Setup:** Qwen3-14B-Base student, 1 seed, trained on all 67 languages (full train split) for the `translated` arm (submitted now) and the `method` arm (after labelling). Recipe and evaluation sets are the same as the core runs.
- **Holds if:** method − translated has the same sign when trained on all 67 as when trained on the core 24, on the core test set and on the 42 extension languages.
- **Reported:** per-language counts of examples dropped by the 1,024-token limit. Scripts with long Qwen tokenisations (e.g. Odia) lose more.

## Amendment 4 (2026-10-10, before labelling): labels and training arms
- **Labels:** the record method (icm/method, RECORD rules, every function unchanged) is run separately per layer:
  - core layer: the jury is the core 24 + en only, so the core dataset stands alone;
  - all-67 layer: the jury is all 67 languages.
  E2 compares the two sets of labels on the core languages.
- **Snapshot:** labels are computed on a frozen snapshot of the Qwen3-14B-Base record (copied before any re-merge).
- **Arms:** each arm covers every pool for which it has a usable completion; coverage per arm and language is reported.
  - In q09, a pool with no gold-correct chain kept the method chain. Here such pools are left out of gold_filtered.
  - Headline comparisons are repeated on the pools every arm covers.
- **Record status:** the Qwen record is used as generated. Empty-output rates match the 2026-08 record, e.g. uz 35%, sw 17%, eu 14%. Any later regeneration under an end-of-text rule would require relabelling.

## Amendment 5 (2026-10-12, implementation only): GPU layout of the remaining trainings
- **What changed:** the 15 trainings still queued on Oct 12 run on 2 GPUs instead of 4, to escape the whole-node queue.
  - Gradient accumulation goes from 2 to 4, so the effective batch stays 32 and every other setting is unchanged.
- **Already done on 4 GPUs:** wave 1; Qwen wave 2; Mistral own_unlabelled s1/s2 and gold_filtered s1.
- **Accounts:** Mistral wave 3 runs on the def-zhijing GPU account (advisor's request); everything else on rrg-zhijing.
- **Reported:** the GPU layout of every run, in the appendix. Seed-to-seed spread is the yardstick for any layout effect.

## Amendment 6 (2026-10-12, implementation only): regeneration context window for the 67-language labels
- **Problem:** with all 67 languages, some regeneration prompts exceed the method's 4,096-token window. The question, two format examples and three peer solutions are chosen under a *character* cap, and long-script languages such as Odia tokenise into many more tokens. The core-25 run never hit this.
- **Fix:** the all-67 regeneration runs with a 16,384-token window. Prompts, peers, sampling and the decision rule are unchanged.
- **Core-25 labels:** unaffected; all their prompts fit in 4,096.
