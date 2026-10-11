# Full Experimental Results — Multilingual ICM Project
## Generation campaign + SM-ICM campaign (2026-07-17 → 2026-07-28)

Model: Qwen/Qwen3-8B-Base (rev 49e3418f) · Dataset: open-multilang/gsm8k-106, train split,
first 50 items/language (parallel across languages) · Cluster: Rorqual H100s.
Companion docs: `pilot_changes.md` (all code changes + reasons), `results_inventory.md`
(file locations + metric definitions). Evaluation always post-hoc against gold; all
generation/selection was gold-blind.

---

# PART A — GENERATION CAMPAIGN

## A1. 6-shot pilot (rorqual-pilot; af am cs en hi; T 0.2/0.7/1.0, n=4/4/2, 512 tok)

| lang | items | usable | correct/incorrect | note |
|---|---|---|---|---|
| af | 50 | 31 | 23/8 | |
| am | 50 | **0** | — | all 50 prompts overlong (3944 tok > 3584 budget) |
| cs | 50 | 98 | 72/26 | |
| en | 50 | 46 | 46/0 | zero incorrect traces |
| hi | 50 | **0** | — | all 50 prompts overlong (4080 tok) |

(usable counts above are from the ORIGINAL quality filter, later shown to be broken — see B3.)

Per-item MIX% (item has ≥1 correct AND ≥1 incorrect usable): af 38%, cs 68%, en 26%,
hi/am 0. ALL = 26%.

**Findings:** (1) 6 native demos overflow the 4096-token context for dense scripts →
hi/am produce nothing. (2) English pools are all-correct → no ICM contrast.

## A2. 6-shot + adaptive retry v1

Retry triggers (label-blind): <4 usable → T=0.2 n=8; <2 distinct answers → T=1.1 n=8.
MIX%: af 38→88, cs 68→92, en 26→64. ALL 26→**49%**. usable/item en 9.1→21.6.
hi/am unchanged (retry manifest was also 6-shot/4096 → skipped again).

## A3. Zero-shot arm (rorqual-zeroshot; n=8/8/4, 512 tok)

Raw MIX% ALL = 76% — but see B-section: heavily inflated by artifacts.
am inverted: 13 correct vs 166 "incorrect" usable (later shown mostly artifact).
hi: 970 raw → 18 pipeline-usable (2%): parse collapse without format examples.

## A4. 2-shot arm (rorqual-2shot; template truncation change)

Prompts fit ALL languages (hi 4080→1620 tok, am 3944→1613). hi/am generate for the
first time. Raw MIX%: hi 76%, am 56%, ALL 55%.

## A5. Honest mix re-analysis (artifact-corrected, sample-matched)

Excluding truncated + last-number-fallback extractions, 0.5% rel tolerance, matched
10 samples/item (0-shot had sampled 20 vs few-shot 10):

| lang | 6-shot | 2-shot | 0-shot |
|---|---|---|---|
| en | 24% | 24% | **68%** |
| cs | 66% | 70% | 74% |
| af | 34% | 38% | 50% |
| hi | — | **56%** | 40% |
| am | — | **20%** | 6% |

**Finding:** 0-shot's diversity gain is REAL for high-resource languages and FAKE
(artifact-driven) for weak ones, where 2-shot wins.

## A6. Stop-sequence arms (rorqual-2shot-stop / rorqual-0shot-stop)

Template-derived per-language stops (native question label + `<example>`).
Truncation: en 12→2, cs 83→46 (0-shot); but am 826→708, hi 254→252 — marker-free
babble is immune. usable counts ~unchanged. New minor issue: 89/117 am 0-shot
stop-fires occurred before any `####` (premature stops on degenerate openings).
**Finding:** stops clean up finished traces; they do not rescue degeneration.

## A7. hi/am rescue (rorqual-hiam-rescue; 2-shot, T 0.2/0.4/0.6, n=8/8/4, 768 tok, stops)

hi: truncation 14/863 (**1.6%**), 17.2 parse-usable/item — Hindi generation solved.
am: truncation 452/952 (47%), honest mix 44% (best am arm; prior arms 36/20/6%).
Post-filter-fix usable: hi **825** (665C/160I), am **509** (181C/328I).
am correct-rate 36% of usable — "am can't do math" was mostly artifact stacking.

## A8. best15 scale-up (rorqual-best15; 15 new langs; 2-shot, T 0.2/0.5/0.8, n=8/8/4, 768, stops)

Job 17002995 crashed (vi trace parsed as Decimal NaN → InvalidOperation in
numbers_match) → parser fix → resume completed in 5 min.

Pre-retry pool MIX% (fixed filter, non-truncated): vi 38, de 38, zh-tw 68, tr 54,
uk 38, el 52, th 62, bn 58, ru 54, ar 70, ko 44, fa 38, kn 58, sw 76, eu 74.
ALL = **55%**. usable/item 8.4–17.9. Truncation ≤4% except kn 29%, sw 12%.

Post-retry-v2: ALL = **75%** (range 56–86). Biggest gains exactly where designed:
uk 38→72, fa 38→72, vi 38→56, de 38→60. Retry added 174–221 candidates for
high-accuracy langs, only 48 for sw.

## A9. View-level mix (what a 4-candidate ICM view actually contains)

random_4 average **43%** mixed vs diverse_4 **78%** (≈ pool level; best15 views —
all-experiment average is 39% vs 66%). Random sampling
destroys half the contrast; diverse_4 (answer-coverage-first, label-blind) preserves it.

## A10. Zero-shot budget experiments (rorqual-0shot15: 15 best15 langs @512 → rorqual-0shot768: all 20 langs @768)

Truncation @512: kn 70%, bn 37%, sw 27%, el 23% → @768: kn 49%, bn 14%, sw 22%, el 9%.
am 69% at 768 (degeneration, budget-insensitive). ALL 15%→11%.

2-shot vs 0-shot768 pools (usable/item | %correct):
overall 16.0 vs 15.2/item (over the 17 recipe-generated languages: best15 + rescue;
legacy af/cs/en 2-shot pools are smaller, pending regeneration) — equivalent volume. High/mid tier equivalent quality.
Low tier: am 10.2@36% vs 6.6@**13%**; sw 16.6@57% vs 11.2@38%; eu 13.2@61% vs
10.3@42%; kn 17.4@61% vs 15.7@44%.
**Finding:** few-shot demos matter twice — termination (volume) and reasoning quality
of survivors (the dominant effect in weak languages).

---

# PART B — DATA-QUALITY FORENSICS

## B1. Deterministic trace audit (`~/trace_review/audit.jsonl`)

- Extraction paths: am 0-shot 730/784 usable were last-number-FALLBACK (junk);
  hi 0-shot 297/938. Parser self-consistency: 0 disagreements on re-extraction.
- Ge'ez-numeral hypothesis DEAD: 1 of ~1,180 am traces uses Ge'ez numerals.
- Arithmetic audit: `<<a op b=c>>` annotations ~99% correct EVEN IN WRONG chains
  (en 6-shot 131/131 pre-retry; 398/402 post-retry) → errors are setup errors;
  arithmetic checkers have little discriminative power on GSM8K-style traces.
- Tolerance flips: only 6–15 en labels flip at 0.5–1% relative tolerance.

## B2. Manual trace reading (~40 traces; `claude_judgments.md`, `VERIFY_ME.md`)

en stopped-wrong: 7/7 genuine plausible errors (ideal ICM negatives). en fallback
extractions: 2/3 were CORRECT traces mislabeled wrong. hi truncated: only ~1/5
on-track (rest degenerate/wrong-before-cut). am truncation = prompt-echo loops +
babble; flawless correct am traces exist. Language drift observed (hi chain
switching to English). Incorrect-trace decomposition: hi 0-shot 185/295 "wrong" =
truncation artifact; only 88 genuine.

## B3. Quality-filter bug (the big one)

Original filter: word type-token-ratio > 0.45 ⇒ reject ("high_repetition").
Reproduced pipeline usable counts EXACTLY with repo's own annotate_quality → proof.
Rejection rates of RAW candidates: en 0-shot 794/960 (83%), hi rescue 835/863 (96% —
of which 832 parseable, **683 with the CORRECT answer**), cs 52–65%, everything 52–96%.
Cause: GSM8K annotated math + particle-heavy languages legitimately repeat words.
Fix: structural degeneracy check (40+ digit runs / same ≥20-char line ×3 / trigram
uniqueness <0.5) + too_short exemption for ####-bearing texts. Validated: catches
all manually-confirmed degenerates, passes legit terse answers; 22 unit tests green.
All views/analyses rebuilt 2026-07-24 (pre-retry 6-shot snapshot kept unrebuilt).

## B4. Low-resource uplift decomposition (post-retry best15 pools)

| lang | raw model acc | majority vote | ICM-precision (where answered) | coverage |
|---|---|---|---|---|
| am | 8% | 36% | 79% | 14/50 |
| sw | 29% | 74% | 88% | 26/50 |
| eu | 23% | 74% | 94% | 31/50 |
| kn | 32% | 74% | 93% | 29/50 |

rescued(majority-wrong→right)=0 everywhere; value = trust-filtering, not rescue.

---

# PART C — SM-ICM CAMPAIGN (structured-multilingual-icm repo)

All runs: Qwen3-8B rev 49e3418f, 3 seeds, thresholds strict_seed_agreement=0.667,
strict_min_confidence=0.5 (as shipped). Input: label-blind all_usable views
(verified: 0 gold keys in all 16,160 adapted records; 40/40 question-text join checks).

## C1. Smoke (vi ru sw × 50 items; 15 min on 4×H100)

Global answers 32/32 correct where committed (18 NULL). Local: vi 92%, ru 86%,
sw 48%. Positive labels: vi 90/90, ru 86/86, sw 48/54. Strict set: 2 chains (2/2 ✓).
Margin diagnostic: 70% of chain margins exactly 0 (robust averaging cancels unstable
marker readings); where nonzero: P(margin>0)=0.76 correct vs 0.44 wrong chains.

## C2. full20 — main run (20 langs × 50; 1 h 49 m)

Global: 26/26 correct where committed (52% coverage). Zero constraint violations.

Local answers per language (coverage/50 | precision):
af 33|100, cs 28|100, de 44|93, el 40|98, en 26|100, tr 42|90, uk 40|98, vi 46|100,
zh-tw 42|98, ar 36|100, bn 42|90, fa 42|98, hi 43|93, ko 38|97, ru 44|100, th 40|100,
am 14|79, eu 31|94, kn 29|93, sw 26|88.
Positive chain labels: 85–100% precision in every language.
Strict SFT set: **4 chains** of ~1,000 possible (4/4 correct) — see C5.

Wrong-answer forensics (hi): all 3 wrong local answers were honest model-consensus
errors (pool voted 12-vs-7 and 11-vs-5 for the wrong number); one case (hi_38) ICM
picked an 8-vote answer over the 10-vote gold answer.

## C3. mono20 — monolingual ablation (cross_language_weight=0; 1 h 35 m)

| tier | FULL cov/prec | MONO cov/prec | Δprec |
|---|---|---|---|
| high | 341/450 97% | 356/450 97% | 0 |
| mid | 285/350 97% | 284/350 97% | 0 |
| low | 100/200 90% | 98/200 87% | **+3** |

Per-language low tier (FULL vs MONO prec): am 79 vs 73 (+6), eu 94 vs 90, kn 93 vs 90,
sw 88 vs 88. Low-tier wrong answers: 10 vs 13 (−23%); of these, converged-but-wrong 10 vs 9
(mono's other 4 errors were non-converged).
**Correction this forced:** smoke→full sw jump (48→88%) was NOT cross-language
transfer — mono also achieves 88%. Thin (3-language) consensus actively hurt.
Cross-language layer = modest, weak-language-concentrated: +3–6 prec, −23%
low-tier errors, +4 pts refusal quality. Most precision is within-language machinery.

## C4. zeroshot20 — data-recipe arm (0-shot768 pools)

| tier | FULL(2shot) | ZEROSHOT |
|---|---|---|
| high | 341/450 97% | 301/450 98% |
| mid | 285/350 97% | 239/350 98% |
| low | 100/200 90% | **28/200 100%** |

Low tier: am 1/50, sw 4/50, kn 6/50, eu 17/50 — all answered items correct.
**Finding:** fail-safe behavior demonstrated — measured pool degradation (13–44%
correct) converts to abstention, not error. Precision invariant; coverage pays.

## C5. Strict-gate threshold sweep (export-stage, exact gating replication)

| seed_thr | conf_thr | strict chains | coverage | precision | low-tier n@prec |
|---|---|---|---|---|---|
| 0.667 | 0.50 (shipped) | 4 | 0.4% | 100% | 1@100% |
| 0.667 | 0.30 | 43 | 4% | 98% | 9@89% |
| 0.500 | 0.10 | 209 | 21% | 97% | 33@91% |
| **0.500** | **0.00** | **606** | **61%** | **96%** | **86@91%** |
| 0.000 | 0.00 | 721 | 72% | 96% | 99@91% |

Cause: selection_confidence = seed_agreement × prompt_stability × sigmoid(margin) —
a product of three [0,1] factors with realistic ceiling ~0.4 (stability median 0.35,
margin median 0); threshold 0.5 was structurally unsatisfiable. Confidence remains
the per-item ranking key at threshold 0. Recommendation: 0.5/0.0 (150× more SFT data
for −4 precision points). NOTE: sweep used gold ⇒ this pilot = development set;
thresholds must be pre-registered before the full-split run.

## C6. Convergence / agreeability

Mean seed agreement ~0.98 in ALL tiers; 97/100 committed low-tier answers have perfect
3-seed agreement. Truth-tracking when converged: high 97%, mid 97%, low 90%.
Converged-but-WRONG (the attractor cell): 10 low-tier items in FULL (am 3, sw 3, eu 2,
kn 2); mono has 9 converged + 4 non-converged. unstable-wrong = **0** in FULL's low
tier (2 mid-tier exceptions at 1/3 agreement: bn_38, hi_38): low-tier errors are
unanimous — internal signals cannot detect the residual failures; only cross-language
evidence (−23% low-tier errors) and abstention do.

## C7. Abstention: mono vs cross-lingual

| tier | run | abst% | correct refusals | over-abstain | refusal quality |
|---|---|---|---|---|---|
| high | FULL/MONO | 24/21% | 20/21 | 90/74 | 18/22% |
| mid | FULL/MONO | 19/19% | 21/20 | 45/47 | 32/30% |
| low | FULL/MONO | 50/51% | 62/59 | 38/43 | **62/58%** |

**Finding:** abstention is a within-language mechanism (rate and targeting ~unchanged
without cross-language evidence). Refusal quality is tier-graded: ~60% of low-tier
NULLs are justified (majority is wrong there) vs ~20% in high tier (conservatism
headroom). In the high tier the cross-language factor slightly over-abstains
(15 extra NULLs, mostly unnecessary, at equal precision).

---

# PART D — CONCLUSIONS & RECOMMENDED CONFIGURATION

1. **Full-split recipe:** 2-shot + max_new_tokens 768 + template-derived stops +
   adaptive retry v2 (`configs/best_regimes.json` + `configs/retry_regimes_v2.json`).
   Chosen on evidence: best coverage of all arms, low-tier pool quality double
   0-shot's, diversity gap closed by retry and largely moot for SM-ICM's objective.
2. **SM-ICM works across the capability gradient:** 79–100% local precision,
   85–100% chain-label precision, all 20 languages, fully gold-blind.
3. **Its character:** near-perfect verifier + trust filter; most precision from
   within-language machinery; cross-language layer contributes modest gains
   concentrated in the weakest languages; abstention converts data poverty into
   silence rather than error.
4. **The one config fix needed:** strict_min_confidence 0.5→0.0 (with seed 0.5) —
   to be pre-registered with the advisor before the full-split run.
5. **Known open items:** correlated-error calibration (runner-up-agreement null
   proposed); ablation `extends` bug in SM-ICM preflight (flat-config workaround
   in use); regenerate 5 legacy languages under best_regimes for provenance
   uniformity; full-split SM-ICM inference ≈ 45–50 h ⇒ multi-allocation resume;
   kn/am truncation-survivor length-selection bias (footnote in any writeup);
   10 attractor items available as qualitative case studies.
