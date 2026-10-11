from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from answer_parsing import (  # noqa: E402
    extract_prediction_answer,
    normalize_number,
    numbers_match,
)
from candidate_processing import (  # noqa: E402
    QualityConfig,
    annotate_quality,
    is_degenerate_text,
)
from prompts import (  # noqa: E402
    available_prompt_names,
    derive_stop_sequences,
    load_prompt_templates,
    render_prompt,
    truncate_example_blocks,
)


TERSE_HINDI = (
    "नतालिया ने मई में 48 / 2 = <<48/2=24>>24 क्लिप्स बेची।\n"
    "कुल मिलाकर 48 + 24 = <<48+24=72>>72 क्लिप्स बेची।\n"
    "#### 72"
)
LEGIT_ENGLISH = (
    "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
    "She makes 9 * 2 = $<<9*2=18>>18 every day at the market.\n"
    "#### 18"
)
ECHO_LOOP = (
    "ጥያቄ፦ ናታሊያ በሚያዚያ ወር ለ48 ጓደኞቿ ክሊፖች ሸጣ ስንት ናቸው?\n"
    "መልስ፦ በግንቦት ወር ናታሊያ ሸጣ 48 - 12 = 36 ክሊፖች ናቸው።\n"
) * 3
DECIMAL_SPAM = "पहले जहाज पर 847 / 3 = <<847/3=282." + "3" * 400 + ">>282.3"
BABBLE = "the model keeps saying the same words over and over " * 5


class NanGuardTests(unittest.TestCase):
    """answer_parsing: non-finite values must never become predictions."""

    def test_nan_and_infinity_rejected(self) -> None:
        for bad in ("NaN", "nan", "Infinity", "-inf", "inf"):
            self.assertIsNone(normalize_number(bad), bad)

    def test_normal_numbers_still_parse(self) -> None:
        self.assertEqual(normalize_number("72"), Decimal("72"))
        self.assertEqual(normalize_number("9.96"), Decimal("9.96"))
        self.assertEqual(normalize_number("1,234"), Decimal("1234"))
        self.assertEqual(normalize_number("-5"), Decimal("-5"))
        self.assertEqual(normalize_number("5/6"), Decimal(5) / Decimal(6))

    def test_nan_final_answer_does_not_crash_scoring(self) -> None:
        raw, parsed = extract_prediction_answer("some reasoning\n#### NaN")
        self.assertIsNone(parsed)
        # the crash path: numbers_match(Decimal("NaN"), gold) raised InvalidOperation
        self.assertFalse(numbers_match(parsed, Decimal("10"), Decimal("0")))


class DegeneracyFilterTests(unittest.TestCase):
    """candidate_processing: catch real degenerates, spare legitimate answers."""

    def test_real_failure_modes_are_caught(self) -> None:
        self.assertTrue(is_degenerate_text(ECHO_LOOP), "prompt-echo loop")
        self.assertTrue(is_degenerate_text(DECIMAL_SPAM), "decimal spam")
        self.assertTrue(is_degenerate_text(BABBLE), "trigram babble")

    def test_legitimate_answers_pass(self) -> None:
        self.assertFalse(is_degenerate_text(TERSE_HINDI))
        self.assertFalse(is_degenerate_text(LEGIT_ENGLISH))

    def test_terse_hindi_survives_full_quality_check(self) -> None:
        # the old TTR filter rejected 835/863 of these as "high_repetition"
        annotated = annotate_quality(
            {"text": TERSE_HINDI, "pred_number": "72"}, QualityConfig()
        )
        self.assertTrue(annotated["quality_ok"], annotated["quality_reasons"])

    def test_too_short_exempts_texts_with_answer_marker(self) -> None:
        short_with_answer = {"text": "48/2 = 24\n#### 72", "pred_number": "72"}
        short_without = {"text": "just some words", "pred_number": "7"}
        self.assertNotIn(
            "too_short",
            annotate_quality(short_with_answer, QualityConfig())["quality_reasons"],
        )
        self.assertIn(
            "too_short",
            annotate_quality(short_without, QualityConfig())["quality_reasons"],
        )

    def test_degenerate_candidates_are_rejected_end_to_end(self) -> None:
        annotated = annotate_quality(
            {"text": ECHO_LOOP, "pred_number": "36"}, QualityConfig()
        )
        self.assertFalse(annotated["quality_ok"])
        self.assertIn("degenerate_text", annotated["quality_reasons"])


class TwoShotTemplateTests(unittest.TestCase):
    """prompts: 2-shot truncation must be clean and 6-shot must be untouched."""

    DEMOS_2 = [
        {"question": "Q1", "answer": "A1\n#### 1"},
        {"question": "Q2", "answer": "A2\n#### 2"},
    ]

    def test_truncation_keeps_exactly_two_blocks(self) -> None:
        for lang in ("en", "hi", "am", "cs", "af"):
            prompt = render_prompt(
                "the question", "few_shot_cot", language=lang,
                demonstrations=self.DEMOS_2,
            )
            self.assertEqual(prompt.count("<example>"), 2, lang)
            self.assertEqual(prompt.count("</example>"), 2, lang)
            self.assertNotRegex(prompt, r"\{demo_\d+_(question|answer)\}", lang)
            self.assertIn("Q2", prompt)

    def test_six_shot_template_is_byte_identical_to_original_path(self) -> None:
        template = load_prompt_templates("few_shot_cot")["en"]
        self.assertEqual(truncate_example_blocks(template, 6), template)

    def test_unsupported_count_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            render_prompt(
                "q", "few_shot_cot", language="en",
                demonstrations=self.DEMOS_2 + [{"question": "Q3", "answer": "A3"}],
            )


class TwoShotWordingTests(unittest.TestCase):
    """few_shot_cot_2shot: fixed languages must say 'two', not 'six'."""

    DEMOS_2 = [
        {"question": "Q1", "answer": "A1\n#### 1"},
        {"question": "Q2", "answer": "A2\n#### 2"},
    ]
    SIX_WORDS = {"en": "six", "hi": "छह", "am": "ስድስት", "ru": "шесть", "de": "sechs",
                 "es": "seis", "pl": "sześć", "ml": "ആറ്", "ceb": "unom"}
    NOUN_STILL_THERE = {"en": "examples", "hi": "उदाहरणों", "am": "ምሳሌዎች",
                        "ru": "примеры", "de": "Beispiele", "es": "ejemplos",
                        "pl": "przykłady", "ml": "ഉദാഹരണങ്ങൾ", "ceb": "pananglitan"}

    def test_fixed_languages_have_count_removed(self) -> None:
        for lang, six in self.SIX_WORDS.items():
            prompt = render_prompt(
                "q", "few_shot_cot_2shot", language=lang,
                demonstrations=self.DEMOS_2,
            )
            header = prompt.split("<example>")[0]
            self.assertNotIn(six, header, lang)
            self.assertIn(self.NOUN_STILL_THERE[lang], header, lang)
            self.assertEqual(prompt.count("<example>"), 2, lang)

    def test_original_template_untouched(self) -> None:
        header = load_prompt_templates("few_shot_cot")["en"].split("<example>")[0]
        self.assertIn("six", header)

    def test_v2_manifests_reference_fixed_template(self) -> None:
        import json

        for manifest in ("configs/best_regimes_v2.json", "configs/retry_regimes_v3.json"):
            config = json.loads((SCRIPTS_DIR.parent / manifest).read_text())
            self.assertEqual(
                config["defaults"]["prompt_template"], "few_shot_cot_2shot", manifest
            )
            self.assertEqual(config["defaults"]["few_shot_count"], 2, manifest)


class StopSequenceTests(unittest.TestCase):
    """prompts: stops must carry the native question label for every language."""

    def test_known_native_labels(self) -> None:
        cases = {
            ("few_shot_cot", "hi"): "\nप्रश्न:",
            ("few_shot_cot", "am"): "\nጥያቄ፦",
            ("zero_shot_cot", "am"): "\nጥያቄ:",
            ("few_shot_cot", "af"): "\nVraag:",
        }
        for (template, lang), expected in cases.items():
            stops = derive_stop_sequences(template, language=lang)
            self.assertIn(expected, stops, (template, lang))
            self.assertIn("<example>", stops)

    def test_all_languages_yield_stops(self) -> None:
        for template in ("few_shot_cot", "zero_shot_cot"):
            for lang in load_prompt_templates(template):
                stops = derive_stop_sequences(template, language=lang)
                self.assertGreaterEqual(len(stops), 2, (template, lang))
                self.assertNotIn("", stops)

    def test_english_prompt_only_uses_english_label(self) -> None:
        stops = derive_stop_sequences(
            "few_shot_cot", language="am", english_prompt_only=True
        )
        self.assertIn("\nQuestion:", stops)


class RunExperimentForwardingTests(unittest.TestCase):
    """run_experiment: manifest flag forwarding for stop sequences."""

    def test_no_stop_flag_forwarded_only_when_disabled(self) -> None:
        import run_experiment

        cmd: list[str] = []
        run_experiment.append_optional_flags(cmd, {"use_stop_sequences": False}, {})
        self.assertIn("--no-use-stop-sequences", cmd)

        cmd = []
        run_experiment.append_optional_flags(cmd, {}, {})
        self.assertNotIn("--no-use-stop-sequences", cmd)


if __name__ == "__main__":
    unittest.main()
