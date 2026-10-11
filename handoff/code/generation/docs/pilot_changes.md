# Code changes since initial clone (pilot experiments, 2026-07-17 → 2026-07-22)

All changes are additive: the original 6-shot pilot behavior is preserved and reproducible.
Author: Suryatej + Claude (pair-debugging during the pilot runs).

## scripts/prompts.py

1. Added `SUPPORTED_FEW_SHOT_COUNTS = (2, 6)` and relaxed the `render_prompt` guard
   (was: exactly 6 demonstrations).
   *Why: enable a 2-shot arm — 6 fully-native demos overflow the 3584-token prompt
   budget for dense scripts (hi 4080, am 3944 tokens), producing zero traces.*

2. Added `truncate_example_blocks()` — drops trailing `<example>` blocks when fewer
   than 6 demonstrations are supplied (all 106 templates verified structurally identical).
   *Why: reuse the existing localized templates for 2-shot without hand-editing 106 files.*

3. Added `derive_stop_sequences()` — derives per-language stop strings (`<example>` +
   the native question label, e.g. `ጥያቄ፦`, `प्रश्न:`) from the template itself.
   *Why: base models keep sampling after the answer, echoing the prompt format until the
   token budget dies; template-derived stops are language-agnostic (no 106-label hardcoding).*

4. Added `import re` and `EXAMPLE_BLOCK_PATTERN` in support of the above.

## scripts/generate_chains.py

5. Guard for `few_shot_cot` now accepts `--few-shot-count` in {2, 6} (was: exactly 6).
   *Why: same as change 1 — unblock the 2-shot arm.*

6. Added `--use-stop-sequences` flag (BooleanOptionalAction, default ON) and the
   `use_stop_sequences` field in `GenerateArgs`.
   *Why: make the stop-sequence behavior explicit, recordable in run_config, and
   disableable per run.*

7. `SamplingParams` is now built per language inside the language loop with
   `stop=derive_stop_sequences(...)` (was: one global SamplingParams with no stops);
   added `build_stop_sequences()` helper.
   *Why: stop strings differ per language; without any stops, 84% of Amharic samples
   ran to the 512-token limit on prompt-echo loops and were mislabeled as truncated.*

## scripts/run_experiment.py

8. `append_optional_flags` forwards `"use_stop_sequences": false` from a manifest as
   `--no-use-stop-sequences`.
   *Why: keep the manifest as the single source of truth for regime behavior.*

## scripts/candidate_processing.py

9. Replaced the word-level type-token-ratio quality check
   (`repetition_ratio > 0.45` → reason `high_repetition`) with structural degeneracy
   detection `is_degenerate_text()` (reason `degenerate_text`): 40+ digit runs
   (decimal spam), the same ≥20-char line repeated ≥3× (echo loops), or word-trigram
   uniqueness < 0.5 (babble). `repetition_ratio` itself is retained but unused.
   *Why: TTR is miscalibrated for annotated GSM8K math answers — numbers and particles
   legitimately repeat — and was rejecting 52–96% of candidates in every language
   (hi rescue: 835/863 rejected, of which 683 had the CORRECT answer); the structural
   checks still catch all manually verified echo/spam degenerates.*

10. `too_short` (< 80 chars) no longer fires when the text contains a `####` answer.
    *Why: terse but complete low-temperature answers were being discarded.*

11. Added `from collections import Counter` and `DIGIT_RUN_RE` in support of change 9.

## scripts/answer_parsing.py

11b. `normalize_number` now rejects non-finite Decimals (`NaN`, `Infinity`).
    *Why: a Vietnamese high-temp trace answered with text parsing to `Decimal("NaN")`,
    and `abs(NaN - gold)` in `numbers_match` raises `decimal.InvalidOperation`,
    crashing an entire generation shard (job 17002995, 2026-07-22).*

## New config files

12. `configs/two_shot_regimes.json` — identical to `pilot_regimes.json` but
    `few_shot_count: 2` on all three temperature regimes.
    *Why: manifest for the 2-shot arm.*

13. `configs/hi_am_rescue_regimes.json` — 2-shot, temperatures 0.2/0.4/0.6,
    `max_new_tokens: 768`, n=8/8/4.
    *Why: hi/am degeneration is temperature-sensitive and hi derivations are verbose;
    low temps + larger budget + stops recovered hi to 825 usable traces (from 13).*

14. `configs/best_regimes.json` — the synthesized final recipe: 2-shot, T 0.2/0.5/0.8,
    n=8/8/4, `max_new_tokens: 768`, stops on.
    *Why: combines every validated fix; used for the 15-language scale-up (rorqual-best15)
    and recommended for the full-split run.*

15. `configs/retry_regimes_v2.json` — adaptive retry at 2-shot / 768 tokens, diversity
    trigger at T=1.0 (was 1.1).
    *Why: retry v1 was 6-shot/4096/512 and silently skipped hi/am (overlong prompts);
    T=1.1 was inside the degeneration zone for weak languages.*

16. `configs/zero_shot_768_regimes.json` — zero-shot manifest with `max_new_tokens: 768`.
    *Why: budget parity with the 2-shot arm for the SM-ICM data-recipe comparison;
    at 512 the 0-shot arm truncated up to 70% (kn) purely from the tighter budget.*

## Not changed (known issues documented but left as-is)

- `answer_parsing.py` last-number fallback (P4) can fabricate predictions from
  unfinished text and mislabel correct traces; excluded analytically, not yet fixed.
- `tolerance: "0"` (exact match) scores rounding near-misses (e.g. 9.96 vs 10) as
  incorrect; a small relative tolerance is recommended for evaluation.
- 2-shot prompts still say "six examples" in the localized headers (known cosmetic
  confound).
- `configs/retry_regimes.json` still uses 6-shot / 4096 and therefore skips hi/am.

## Post-fix rebuilds (no code, for the record)

- After change 9/10, `build_candidate_views.py` was re-run (`--overwrite`) for:
  rorqual-pilot, rorqual-2shot, rorqual-2shot-stop, rorqual-zeroshot,
  rorqual-0shot-stop, rorqual-hiam-rescue. The pre-retry backup
  `rorqual-pilot/final_before_retry` was intentionally left untouched.
- All 7 unit tests in `tests/test_candidate_views.py` pass after every change.
