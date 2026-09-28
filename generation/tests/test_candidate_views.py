from __future__ import annotations

import sys
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from candidate_processing import (  # noqa: E402
    QualityConfig,
    candidate_statistics,
    prepare_candidates,
    sanitize_record,
    select_diverse,
    select_oracle_balanced,
    select_random,
)
from build_candidate_views import process_language  # noqa: E402
from run_adaptive_retry import should_retry  # noqa: E402


def candidate(index: int, answer: str, correct: bool, regime: str = "mid") -> dict:
    reasoning = " ".join(f"step{index}_{word}" for word in range(30))
    return {
        "global_sample_id": f"run:{regime}:{index}",
        "regime_name": regime,
        "candidate_source": "main",
        "sample_index": index,
        "text": f"{reasoning}\n#### {answer}",
        "pred_number": answer,
        "correct": correct,
    }


class CandidateViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            candidate(0, "10", True, "low"),
            candidate(1, "10", True, "mid"),
            candidate(2, "11", False, "mid"),
            candidate(3, "12", False, "high"),
            candidate(4, "13", False, "high"),
        ]
        self.annotated, self.usable = prepare_candidates(
            self.candidates, QualityConfig(min_chars=10)
        )

    def test_statistics_keep_correctness_for_analysis(self) -> None:
        stats = candidate_statistics(
            {"id": "train_en_0", "language": "en", "gold_number": "10"},
            self.annotated,
            self.usable,
        )
        self.assertEqual(stats["correct_usable_count"], 2)
        self.assertEqual(stats["incorrect_usable_count"], 3)
        self.assertEqual(stats["distinct_predicted_answers"], 4)

    def test_sanitized_icm_record_has_no_gold_or_correctness(self) -> None:
        output = sanitize_record(
            {
                "id": "train_en_0",
                "question": "Question",
                "gold_number": "10",
                "gold_raw": "#### 10",
                "original_answer": "answer",
                "candidates": self.candidates,
            },
            self.usable,
        )
        self.assertNotIn("gold_number", output)
        self.assertNotIn("gold_raw", output)
        self.assertNotIn("original_answer", output)
        self.assertTrue(all("correct" not in item for item in output["candidates"]))

    def test_random_selection_is_deterministic_per_item(self) -> None:
        first = select_random(self.usable, 4, 1729, "train_en_0")
        second = select_random(self.usable, 4, 1729, "train_en_0")
        self.assertEqual(
            [item["global_sample_id"] for item in first],
            [item["global_sample_id"] for item in second],
        )

    def test_diverse_selection_covers_distinct_predicted_answers(self) -> None:
        selected = select_diverse(self.usable, 4, 1729, "train_en_0")
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({item["pred_number"] for item in selected}), 4)

    def test_oracle_balanced_selection_is_explicit(self) -> None:
        selected = select_oracle_balanced(self.usable, 4, min_correct=1, max_correct=2)
        self.assertEqual(len(selected), 4)
        self.assertEqual(sum(item["correct"] for item in selected), 2)

    def test_language_artifacts_separate_views_from_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "pooled" / "generated" / "en.jsonl"
            input_path.parent.mkdir(parents=True)
            record = {
                "id": "train_en_0",
                "language": "en",
                "question": "What is the answer?",
                "gold_number": "10",
                "gold_raw": "#### 10",
                "candidates": self.candidates,
            }
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            args = Namespace(
                output_dir=root,
                views=["all_usable", "random_4", "diverse_4"],
                overwrite=True,
                include_gold_in_views=False,
                size=4,
                seed=1729,
                min_correct=1,
                max_correct=2,
            )
            process_language(args, QualityConfig(min_chars=10), input_path)

            view = json.loads(
                (root / "candidate_views" / "all_usable" / "en.jsonl").read_text()
            )
            stats = json.loads(
                (root / "analysis" / "item_candidate_stats" / "en.jsonl").read_text()
            )
            self.assertNotIn("gold_number", view)
            self.assertTrue(all("correct" not in item for item in view["candidates"]))
            self.assertEqual(stats["correct_usable_count"], 2)

    def test_retry_triggers_are_label_blind(self) -> None:
        stats = {
            "usable_candidate_count": 3,
            "distinct_predicted_answers": 1,
            "correct_usable_count": 3,
        }
        self.assertTrue(
            should_retry(
                {"retry_when": "insufficient_usable", "min_usable": 4}, stats
            )
        )
        self.assertTrue(
            should_retry(
                {"retry_when": "low_answer_diversity", "min_distinct_answers": 2}, stats
            )
        )
        with self.assertRaises(ValueError):
            should_retry({"retry_when": "no_correct"}, stats)


if __name__ == "__main__":
    unittest.main()
