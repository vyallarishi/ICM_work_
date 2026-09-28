# Results Inventory — Multilingual ICM Pilot (2026-07-17 → 2026-07-25)

Model: Qwen/Qwen3-8B-Base (vLLM). Dataset: open-multilang/gsm8k-106, train split,
first 50 items per language (same `original_id`s in every language — parallel data).
All paths below: `$SCRATCH = /lustre10/scratch/rvyalla` (a.k.a. `~/links/scratch`),
`$REPO = /home/rvyalla/multilingual_icm`. Results root: `$SCRATCH/multilingual-icm/results/`.

## 1. Experiment registry

| Experiment dir | Config | Languages | Slurm job(s) |
|---|---|---|---|
| `rorqual-pilot` | 6-shot, T 0.2/0.7/1.0, n=4/4/2, 512 tok | af am cs en hi | 16635058; retry 16677775 |
| `rorqual-pilot/final_before_retry` | backup of 6-shot BEFORE retry (pre-retry snapshot, untouched) | same | — |
| `rorqual-zeroshot` | 0-shot, T 0.2/0.7/1.0, n=8/8/4, 512 tok | same 5 | 16677658 |
| `rorqual-2shot` | 2-shot, same temps as pilot, 512 tok | same 5 | 16677701 |
| `rorqual-2shot-stop` / `rorqual-0shot-stop` | same + stop sequences | same 5 | 16799476 / 16799481 |
| `rorqual-hiam-rescue` | 2-shot, T 0.2/0.4/0.6, n=8/8/4, **768 tok**, stops | am hi | 16900697 |
| `rorqual-best15` | `configs/best_regimes.json`: 2-shot, T 0.2/0.5/0.8, n=8/8/4, 768 tok, stops; + retry v2 | vi de zh-tw tr el th bn ru ar ko fa kn sw eu uk | 17002995 (crashed: NaN bug), 17006490 (resume, COMPLETED), retry 17008423 |

Configs: `$REPO/configs/{pilot,two_shot,zero_shot,hi_am_rescue,best}_regimes.json`,
`retry_regimes.json` (v1, 6-shot — silently skips hi/am), `retry_regimes_v2.json` (2-shot/768).
All code changes + reasons: `$REPO/docs/pilot_changes.md`.

## 2. Primary data files (per experiment)

- Pooled candidates WITH gold + correct flags: `<exp>/final/pooled/generated/<lang>.jsonl`
  (one JSON line per item: question, gold_number, candidates[] with text, pred_number,
  correct, finish_reason, stop_reason, temperature, regime_name, candidate_source)
- Raw per-regime (immutable, includes rendered prompt + prompt_token_count):
  `<exp>/workers_2gpu/shard_<0|1>/runs/<regime>/generated/<lang>.jsonl`
- Pipeline stats: `<exp>/final/analysis/candidate_stats_by_language.csv` and
  `<exp>/final/analysis/item_candidate_stats/<lang>.jsonl` (per-item, incl. quality_failure_counts)
- Label-blind ICM inputs: `<exp>/final/candidate_views/{all_usable,random_4,diverse_4}/<lang>.jsonl`
- NOTE: all `final/` analyses+views were REBUILT on 2026-07-24 with the FIXED quality filter.
  `rorqual-pilot/final_before_retry` was NOT rebuilt (kept as historical snapshot).

## 3. Every result computed, with method

### R1. Per-language generation stats (all arms)
From `candidate_stats_by_language.csv`. Pre-filter-fix values are OBSOLETE (see R8).
Key post-fix usable counts (usable = quality-pass minus dupes): rescue hi 825 (665C/160I),
rescue am 509 (181C/328I), 0shot-stop en 896, hi 893, am 435; 2shot hi 465.

### R2. Per-item mix classification (the core metric)
Method: per item, usable = candidates with pred_number != None; item is MIXED if
>=1 correct AND >=1 incorrect usable; else all_correct / all_incorrect / none.
mix% = MIXED items / 50. Computed by inline Python over pooled files.
6-shot: af 38% cs 68% en 26% (hi, am = 0 traces, overlong prompts). ALL 26%.
6-shot+retry(v1): af 88% cs 92% en 64% → ALL 49%.
2-shot: hi 76% am 56% ALL 55%. 0-shot: ALL 76% (en 80%).

### R3. Honest mix% (artifact-corrected)
Method: exclude finish_reason=="length" (truncated) and last-number-fallback
extractions (no #### or answer-line in text); correct at 0.5% RELATIVE tolerance.
0-shot hi 82%→50%, am 44%→6%; en 80%→78%. Shows 0-shot "diversity win" for hi/am was artifact.

### R4. Honest + sample-matched mix% (fair cross-arm comparison)
Method: R3 + cap each item at 10 candidates matched by temperature bucket (4 low/4 mid/2 high),
because 0-shot sampled 20/item vs 10 for few-shot arms.
en: 6shot 24% = 2shot 24% << 0shot 68% (REAL gain); hi: 2shot 56% > 0shot 42%;
am: 2shot 20% > 0shot 6% (0-shot "gain" was fake for weak langs).
distinct-answers/item also computed here (label-free diversity).

### R5. Deterministic trace audit ("Phase A") — persisted at `~/trace_review/audit.jsonl`
Per-candidate rows (arm, lang, item, pred, correct, finish, extraction path, parser_agree,
annotation counts, tolerance flips). Methods + findings:
- Extraction path classes P1 clean-####/P2 messy/P3 answer-line/P4 last-number-fallback.
  am 0-shot: 730/784 usable are P4 (junk extractions); hi 0-shot 297/938.
- Parser self-consistency: re-ran repo extract_prediction_answer on all texts: 0 disagreements.
- Ge'ez/Devanagari numeral scan: 1 trace of ~1180 am uses Ge'ez numerals → parser-script
  hypothesis DEAD; model writes ASCII digits.
- Arithmetic audit: evaluated all <<a op b = c>> annotations (safe eval, 1% rel tol):
  ~99% arithmetically correct EVEN IN WRONG CHAINS (en 6shot 444/444) → errors are
  setup errors, not calculation errors; arithmetic checkers have near-zero discrimination here.
- Tolerance flips: only 6–15 en labels flip at 0.5–1% rel tolerance (rounding artifacts, e.g. 9.96 vs 10).
- Truncation counts per arm/lang (pre-stop-fix): am 359–826, hi 70–254 per arm.

### R6. Manual trace judgments ("Phase B") — `~/trace_review/claude_judgments.md`
~40 traces read and classified (genuine error / arithmetic slip / wrong extraction /
mislabeled-correct / degenerate / language-drift), stratified by bucket files
`~/trace_review/{en,hi,am}_{1_incorrect_truncated,2_incorrect_stopped,3_correct}.txt`.
Findings: en stopped-wrong 7/7 genuine plausible errors; en P4 fallback mislabeled
CORRECT traces 2/3; hi truncated only ~1/5 on-track (rest degenerate/wrong-before-cut);
am truncation driven by prompt-echo loops + babble; genuine am capability exists
(flawless correct traces found); language drift found (hi trace switching to English).
User verification set: `~/trace_review/VERIFY_ME.md` (10 cases).

### R7. Incorrect-trace decomposition (en/hi, all arms)
Method: wrong traces split into truncated / near-miss (0.5% rel) / malformed (no clean
trailing ####) / genuine. hi 0-shot: 185/295 wrong = truncated (63%), only 85 genuine.
en 6-shot: 33/39 genuine, 0 truncated. 0-shot adds 23–24 malformed per lang vs 0 in 6-shot.

### R8. Quality-filter audit + fix validation
Method: ran repo's own annotate_quality over all pooled files; reproduced pipeline
usable counts EXACTLY (proof of cause). OLD filter (word type-token ratio > 0.45 =
"high_repetition") rejected 52–96% of candidates in EVERY language (en 0-shot 798/963;
hi rescue 835/863, of which 832 parseable and 683 CORRECT). NEW filter
(is_degenerate_text: 40+ digit runs / same >=20-char line 3x / trigram uniqueness <0.5)
+ too_short exemption for ####-bearing texts: known degenerates still caught (am echo,
decimal spam), legit terse answers pass. All 7 unit tests pass. Views rebuilt for all
6 experiment dirs (backup untouched).

### R9. Stop-sequence before/after
Method: compare truncation + stop_reason + usable across old vs -stop arms.
Stops fire correctly (native labels e.g. ጥያቄ፦ derived from templates), en/cs truncation
→ ~0, but hi/am truncation barely moved (am 826→708; marker-free babble immune);
usable counts ~unchanged; NEW issue: 89/117 am 0-shot stop-fires happened BEFORE any
#### (premature stops). Conclusion: stops clean finished traces; don't rescue degeneration.

### R10. hi/am rescue evaluation
rescue (T<=0.6, 768 tok, n=20): hi truncation 14/863 (1.6%), 17.2 parse-usable/item;
am best arm ever (honest mix 44% vs 36/20/6% prior arms). Exposed R8 (pipeline said
hi usable=13 while parse-level said 860 → filter bug found).

### R11. best15 scale-up (15 new languages, final pipeline)
Pre-retry per-item mix (usable = non-truncated + quality-pass; method of R2 on fixed filter):
vi 38 de 38 zh-tw 68 tr 54 uk 38 el 52 th 62 bn 58 ru 54 ar 70 ko 44 fa 38 kn 58 sw 76 eu 74;
ALL 55%. usable/item 8.4–17.9. Truncation <=4% except kn 29%, sw 12%.
Post-retry(v2): ALL 75%; biggest gains uk 38→72, fa 38→72, vi 38→56, de 38→60;
retry added 174–221 candidates for high-acc langs, 48 for sw. Range now 56–86% mixed.

### R12. View-level mix (what ICM actually sees) — KEY DESIGN RESULT
Method: for each 4-candidate view record, rejoin gold from pooled by item id;
view is mixed if >=1 pred==gold and >=1 pred!=gold. random_4 average 43% vs
diverse_4 78% (diverse_4 ~= pool-level mix; random sampling destroys half the contrast).
=> diverse_4 is the primary ICM input view.

### R13. NaN crash + fix
Job 17002995 shard 0 crashed: vi trace parsed as Decimal("NaN") → numbers_match
InvalidOperation. Fixed in answer_parsing.normalize_number (reject non-finite).
Resume (same EXPERIMENT_NAME) completed in 5 min.

### R14. External reference data (user-supplied, not in repo)
(a) accuracy_by_language JSONs: prior full-split (1319-item) zero-shot eval per language —
used for language selection tiers. (b) _aggregate_summary.json: translation QC
(judge scores, keep rates) — used to exclude bad-translation languages (zu xh ny ht ha sm
sn so st, hmn haw mi ig <=0.06 keep) and parse-collapse languages (he 95% unparsable,
gd hy ta pa mn te sq tg si yi my 40–95%).

## 4. Key caveats to carry into any report
- Two "usable" definitions exist: parse-level (pred_number != None) vs pipeline
  (quality filter + dedup). Post-fix they largely agree; pre-fix pipeline numbers are
  filter-starved and must not be cited.
- All pre-2026-07-24 candidate_views and analysis CSVs were rebuilt; `final_before_retry`
  is the only pre-fix snapshot (pilot 6-shot).
- Retry rebuilt `rorqual-best15/final` IN PLACE; pre-retry best15 numbers exist only in
  R11 above (no disk snapshot was made).
- Known unfixed issues (documented, excluded analytically): P4 last-number fallback still
  in parser; tolerance "0" in configs (analyses used 0.5% rel); 2-shot header says
  "six examples"; retry v1 manifest is stale (use v2).
- am correctness interpretation: weak-but-real capability; mass "incorrect" was
  echo/truncation/extraction artifact; genuine correct rate ~36% of usable in rescue arm.
- Analyses were inline scripts (not persisted) EXCEPT audit.jsonl + judgment files;
  formulas are fully specified above for reproduction.
