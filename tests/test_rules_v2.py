"""The v2 rules: each fix does what it says, and the RECORD path is untouched.

    python -m unittest tests/test_rules_v2.py
"""
import copy
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icm.method import data, regenerate, vote                         # noqa: E402
from icm.method.metrics import metrics                                 # noqa: E402
from icm.method.rules import RECORD, V2, Rules                         # noqa: E402
from icm.experiments.q07_audits import fix_pool_view_share as fixed    # noqa: E402

C = lambda a, text="t": {"answer": a, "text": text}


def pool(answers, gold="1", lang="en"):
    chains = [C(a, f"text{i}") for i, a in enumerate(answers)]
    counts = Counter(answers)
    top = max(counts, key=lambda a: (counts[a], a))
    return {"gold": gold, "lang": lang, "question": "q", "chains": chains,
            "share": counts[top] / len(answers), "plurality": top}


def synthetic_pools(seed=3, n_problems=40):
    rng = random.Random(seed)
    langs = ["en", "de", "am", "sw", "ja", "my"]
    golds = [str(rng.randint(5, 500)) for _ in range(n_problems)]
    pools = {}
    for lang in langs:
        for k in range(n_problems):
            gold = golds[k]
            wrong = [str(int(gold) + d) for d in (1, 7, -3)]
            p_gold = rng.choice([0.85, 0.5, 0.2])
            answers = [gold if rng.random() < p_gold else rng.choice(wrong) for _ in range(rng.randint(3, 14))]
            pools[f"train_{lang}_{k}"] = pool(answers, gold, lang)
    return pools


class FixedView(unittest.TestCase):
    def test_equals_the_audited_fix(self):
        pools = synthetic_pools()
        a = data.apply_diverse_views_fixed(copy.deepcopy(pools))
        b = fixed.apply_diverse_views(copy.deepcopy(pools))
        self.assertEqual(a, b)

    def test_true_share_and_order_invariance(self):
        p = pool(["10"] * 10 + ["20"] * 10)
        data.apply_diverse_views_fixed({"x": p})
        self.assertEqual(p["share"], 0.5); self.assertEqual(p["full_size"], 20); self.assertEqual(len(p["chains"]), 8)
        chains = [C(str(i % 5), f"t{i}") for i in range(30)]
        self.assertEqual(sorted(c["text"] for c in data.diverse8_view_fixed(chains)),
                         sorted(c["text"] for c in data.diverse8_view_fixed(list(reversed(chains)))))


class Switches(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(data.match("200", "201"))
        self.assertFalse(data.match("200", "201", V2))
        self.assertTrue(data.match("8", "8", V2))
        self.assertTrue(data.match(data.norm("8.0"), "8", V2))

    def test_veto_by_match(self):
        p = pool(["100", "100", "50"])
        self.assertIsNone(vote.update_pool(p, ["100.4"] * 10))                      # record: string veto
        self.assertEqual(vote.update_pool(p, ["100.4"] * 10, Rules("x", veto_by_match=True)), "100")
        self.assertIsNone(vote.update_pool(p, ["100.4"] * 10, V2))                  # v2: exact -> not in pool
        self.assertEqual(vote.update_pool(p, ["100"] * 10, V2), "100")
        self.assertIsNone(regenerate.weighted_decide(p, ["100.4"] * 10, [], 0.5, False))
        self.assertEqual(regenerate.weighted_decide(p, ["100.4"] * 10, [], 0.5, False,
                                                    Rules("x", veto_by_match=True)), "100")

    def test_units_fallback(self):
        for text in ("#### 42 dollars", "#### $42", "#### 42 apples."):
            self.assertIsNone(regenerate.extract_answer(text))
            self.assertEqual(regenerate.extract_answer(text, V2), "42", text)
        self.assertEqual(regenerate.extract_answer("#### 72", V2), "72")
        self.assertIsNone(regenerate.extract_answer("#### none", V2))

    def test_metrics_exact(self):
        pools = {"a": pool(["201"], gold="200")}
        st = {"a": {"answer": "201"}}
        self.assertEqual(metrics(pools, st, [])["precision"], 1.0)
        self.assertEqual(metrics(pools, st, [], V2)["precision"], 0.0)


class RecordUnchanged(unittest.TestCase):
    """Default arguments must reproduce the record exactly (the equivalence test
    in test_method_matches_experiments.py is the stronger version of this)."""

    def test_defaults_are_record(self):
        pools = synthetic_pools()
        a = vote.converge(copy.deepcopy(pools))
        b = vote.converge(copy.deepcopy(pools), rules=RECORD)
        self.assertEqual(a, b)
        v2 = vote.converge(copy.deepcopy(pools), rules=V2)
        self.assertEqual(len(v2[1]) > 0, True)

    def test_v2_changes_something_on_synthetic_data(self):
        pools = synthetic_pools()
        rec = data.apply_diverse_views(copy.deepcopy(pools))
        fix = data.apply_diverse_views_fixed(copy.deepcopy(pools))
        self.assertNotEqual([p["share"] for p in rec.values()], [p["share"] for p in fix.values()])


if __name__ == "__main__":
    unittest.main()
