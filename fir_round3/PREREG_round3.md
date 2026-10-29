# Pre-registration, round 3 (Fir): controls, the S-rule, a third student, a third generator, flaw tolerance

Written 2026-10-29 before any round-3 set was built or job submitted. Verdicts HOLDS / FAILS against these bars,
no re-interpretation. Prior rounds: `data/native_vs_sft/{hypotheses,floor,gemma}/` (Rorqual).

## Frozen inputs
- Floor set: 38 languages (core 25 + am mn cy km ky | yi ka lo or ml kk gu pa), all 7,473 problems; arms
  `translated`, `own_unlabelled`, `routed` as built on Rorqual (shipped, checksummed).
- Self-consistency thresholds, frozen from the pooled two-generator fit: **Qwen student 0.847, Mistral student
  0.718**; Gemma student uses 0.718 (its base accuracy, 25%, is nearer Mistral's 12% than Qwen's 60%; stated
  here, not tuned). S-routes per student are in `inputs/floor_inputs.json`.
- Random-mix share: 0.184 (the A-routed arm's translated share), chosen per example with a fixed seed.
- Recipe, evaluation: unchanged (core-24 recipe; greedy exact match; `eval_floor.jsonl` 38 × 1,317; MGSM human).

## Tier 1 — floor set, seed 2, students Qwen3-14B-Base, Mistral-Nemo-Base-2407, gemma-3-12b-pt-text
Arms per student: translated, own_unlabelled, routed (A-rule, 7 languages translated), s_routed (S-rule),
random_mix, union (both sources, 2× examples), oracle (Qwen, Mistral only: per-language winner at seed 1).

**R1 (controls; per student, overall accuracy on eval_floor):**
- R1a routed − random_mix ≥ +0.5 (routing, not blending);
- R1b routed − union ≥ −0.5 (routing is not worse than using both sources at twice the data);
- R1c oracle − routed ≤ +1.0 (the label-free rule is within a point of the test-set oracle);
- R1d s_routed − max(own, translated) ≥ +0.5, and s_routed − routed ≥ −0.5.
Prediction: all hold for Qwen and Mistral; R1a and R1d uncertain for Mistral (small margins).

**R2 (seed 2 replicates seed 1):** the signs of F1, F3, F4 (`floor/PREREG_floor.md`) hold at seed 2 for Qwen and
Mistral, and the seed-1/seed-2 difference of every arm's overall accuracy is ≤ 1.0.

**R3 (third student):** for Gemma, Spearman(own − translated, S) ≥ 0.5 over the 37 non-English languages, and
routed − max(own, translated) ≥ +0.5. Prediction: HOLDS; the floor sits lower than Qwen's (weak student).

## Tier 2a — Mistral-NeMo as generator (core 25, 2-shot, record recipe)
Smoke (50 problems/language) must pass `lib/gen_check.py` gates before the full run. Arms from the Mistral
record: `mistral_own_unlabelled`, `mistral_gold_filtered`, `mistral_s_routed_<student>`; plus `translated` and
`qwen_gold_filtered` (from the shipped core-24 files) trained on Fir in the same round, so every comparison is
within one evaluation job. Students: Qwen, Mistral.

**M1 (floor transfers to a third generator):** sign(mistral_own − translated) = sign(S_mistral − threshold) in
≥ 18 of 24 languages, each student; Spearman(gain, S_mistral) ≥ 0.5.
**M2 (on-policy vs writer quality), the Gemma G4 puzzle.** With correctness held at 100%:
- on-policy reading: for the Mistral student mistral_gold − qwen_gold ≥ +1.0 while for the Qwen student it is
  ≤ −1.0 (each student prefers its own writing);
- writer-quality reading: both students prefer the same writer by ≥ 1.0 (same sign);
- neither (both within ±1.0): correctness-matched writer identity does not matter at this scale.
Prediction: unsure; this is the experiment that decides it.
**M3 (descriptive):** mistral_s_routed vs both pure arms; mistral_gold − mistral_own (correctness premium) vs
Qwen's (+2.3 / +3.3) and Gemma's (+13.4 / +11.4).

## Tier 2b — flaw tolerance by competence (Qwen student)
Tiers high (es fr id), mid (tr th tl), low (uz hy te); pools with ≥ 1 correct and ≥ 1 wrong usable chain; mixes
p ∈ {0, 25, 50, 75, 100}% wrong (nested); reference = translated on the same pools; test = the tier's 3
languages × 1,317. 18 trainings.
**T1 (monotone):** per tier, accuracy decreases with p (Spearman over the 5 points ≤ −0.8).
**T2 (the floor from the other side):** per tier, the interpolated p at which the curve crosses the translated
reference. Prediction: the crossing lies at 20–35% wrong in every tier (i.e. a correctness floor of ~65–80%
that does not depend on the tier's resource level). Bar: the three crossings lie within 15 points of each other.
**T3 (descriptive):** p = 0 (all correct) vs translated per tier; p = 100 (all wrong) vs base.

## Decision rules
- R1 (a, c, d) and R2 hold → the routing contribution is defensible against "any mix helps" and "use both".
- M1 holds → the floor is shown on three generators with one frozen rule.
- M2 → whichever reading holds becomes a stated finding; "neither" is also reported.
- T1 and T2 hold → the floor is a correctness property independent of resource level, measured directly.
