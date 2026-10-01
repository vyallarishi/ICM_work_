"""Edge cases for every function that decides, compares or scores.

Each test states what the function SHOULD do. Where the code does something
else, the test is marked `expectedFailure` with a KNOWN ISSUE note, so this
suite doubles as the list of open problems in the deciding logic. Synthetic
inputs only; runs in well under a second.

    python -m unittest tests/test_edge_cases.py -v
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generation" / "scripts"))

from icm.method import data, regenerate, vote                                 # noqa: E402
from icm.method.metrics import metrics                                         # noqa: E402
from icm.experiments.q07_audits import fix_pool_view_share as fixed            # noqa: E402
import answer_parsing as gen                                                   # noqa: E402

C = lambda a, text="t": {"answer": a, "text": text}


def pool(answers, gold="1", lang="en", **kw):
    chains = [C(a, f"text{i}") for i, a in enumerate(answers)]
    from collections import Counter
    counts = Counter(a for a in answers)
    top = max(counts, key=lambda a: (counts[a], a))
    return {"gold": gold, "lang": lang, "question": "q", "chains": chains,
            "share": counts[top] / len(answers), "plurality": top, **kw}


# ---------------------------------------------------------------- norm / match
class Norm(unittest.TestCase):
    def test_canonical_forms(self):
        for raw, want in [("1,200", "1200"), ("1200.0", "1200"), ("0.50", "0.5"), (".5", "0.5"),
                          ("1e3", "1000"), ("  42 ", "42"), ("-7.50", "-7.5"), ("1 200", "1200"),
                          ("١٢٣", "123"), ("१२३", "123")]:
            self.assertEqual(data.norm(raw), want, raw)

    def test_rejects_non_numbers(self):
        for raw in ["42 dollars", "$42", "", None, "nan", "inf", "3/4", "abc", "1" + "0" * 30]:
            self.assertIsNone(data.norm(raw), raw)

    @unittest.expectedFailure
    def test_negative_zero_is_zero(self):
        # KNOWN ISSUE: norm("-0") == "-0", a distinct string from "0"; match() says equal,
        # but every `answer in counts` membership test (the veto) compares strings.
        self.assertEqual(data.norm("-0"), "0")

    @unittest.expectedFailure
    def test_spaces_between_distinct_numbers(self):
        # KNOWN ISSUE: spaces are stripped before parsing, so "1 2" becomes 12.
        self.assertIsNone(data.norm("1 2"))


class Match(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(data.match("8", "8"))
        self.assertTrue(data.match(data.norm("8.0"), data.norm("8")))
        self.assertFalse(data.match(None, "8"))
        self.assertFalse(data.match("-5", "5"))
        self.assertTrue(data.match("0", "0"))
        self.assertFalse(data.match("0", "0.004"))

    @unittest.expectedFailure
    def test_distinct_integers_are_distinct(self):
        # KNOWN ISSUE: the 0.5% RELATIVE tolerance merges distinct integers once they
        # are >= 200: 200 vs 201, 1000 vs 1004, 2000 vs 2010 all count as the same answer.
        # GSM8K answers are integers, so this can score a wrong answer as correct.
        self.assertFalse(data.match("200", "201"))
        self.assertFalse(data.match("1000", "1004"))
        self.assertFalse(data.match("2000", "2010"))


# ------------------------------------------------------------- extract_answer
class ExtractAnswer(unittest.TestCase):
    def test_answer_line(self):
        for text, want in [("#### 72", "72"), ("####72", "72"), ("#### 72.", "72"),
                           ("#### 1,200", "1200"), ("#### -5", "-5"), ("#### 5\n#### 6", "5"),
                           ("blah 3 + 4 = 7 apples", "7"), ("no digits here", None)]:
            self.assertEqual(regenerate.extract_answer(text), want, text)

    @unittest.expectedFailure
    def test_answer_line_with_units(self):
        # KNOWN ISSUE: "#### 42 dollars" / "#### $42" return None (the chain is dropped)
        # instead of falling back to the number in the line.
        self.assertEqual(regenerate.extract_answer("#### 42 dollars"), "42")
        self.assertEqual(regenerate.extract_answer("#### $42"), "42")

    @unittest.expectedFailure
    def test_fraction_and_scientific(self):
        # KNOWN ISSUE: "3/4" -> "4" (last number), "1e5" -> "5". The generation-side
        # parser handles fractions; this one, used for regenerated chains, does not.
        self.assertEqual(regenerate.extract_answer("#### 3/4"), "0.75")
        self.assertEqual(regenerate.extract_answer("#### 1e5"), "100000")


# ------------------------------------------------------------ generation side
class GenerationParser(unittest.TestCase):
    def test_marked_answers(self):
        from decimal import Decimal as D
        for text, want in [("#### 72", D(72)), ("#### $1,200.50", D("1200.50")), ("#### 3/4", D("0.75")),
                           ("The answer is 42.", D(42)), ("#### −5", D(-5)), ("#### 5 or 6", None),
                           ("#### 42 dollars", None)]:
            self.assertEqual(gen.extract_prediction_answer(text)[1], want, text)

    @unittest.expectedFailure
    def test_space_grouped_thousands(self):
        # KNOWN ISSUE: "#### 1 200" (space as thousands separator, common in fr/ru/sv
        # text) parses as 200.
        from decimal import Decimal as D
        self.assertEqual(gen.extract_prediction_answer("#### 1 200")[1], D(1200))

    def test_numbers_match_exact_by_default(self):
        from decimal import Decimal as D
        self.assertTrue(gen.numbers_match(D(5), D("5.0"), D(0)))
        self.assertFalse(gen.numbers_match(D(200), D(201), D(0)))


# ------------------------------------------------------------------ load_pools
class LoadPools(unittest.TestCase):
    def write(self, rows):
        d = tempfile.mkdtemp()
        g = Path(d) / "final" / "pooled" / "generated"; g.mkdir(parents=True)
        (g / "en.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return d

    def test_filters(self):
        cand = lambda p, fin="stop", text="x": {"pred_number": p, "finish_reason": fin, "text": text}
        root = self.write([
            {"id": "train_en_1", "question": "q", "gold_number": "5",
             "candidates": [cand("5"), cand("5", "length"), cand(None), cand("abc"), cand("6", text="  ")]},
            {"id": "train_en_2", "question": "q", "gold_number": None, "candidates": [cand("5")]},
            {"id": "train_en_3", "question": "q", "gold_number": "5", "candidates": [cand(None)]},
        ])
        pools = data.load_pools(root)
        self.assertEqual(list(pools), ["train_en_1"])           # no gold / no usable chain -> dropped
        self.assertEqual([c["answer"] for c in pools["train_en_1"]["chains"]], ["5"])  # 4 of 5 dropped
        self.assertEqual(pools["train_en_1"]["share"], 1.0)

    def test_tie_break_is_deterministic(self):
        p = pool(["9", "10", "10", "9"])
        self.assertIn(p["plurality"], ("9", "10"))

    @unittest.expectedFailure
    def test_tie_break_is_numeric(self):
        # KNOWN ISSUE: ties between equally common answers are broken by STRING order,
        # so "9" beats "10" and "7" beats "60". Deterministic but arbitrary.
        self.assertEqual(pool(["9", "10", "10", "9"])["plurality"], "10")


# ----------------------------------------------------------------------- views
class Views(unittest.TestCase):
    def test_small_pool_unchanged(self):
        chains = [C(str(i)) for i in range(8)]
        self.assertEqual(data.diverse8_view("train_en_1", chains), chains)

    def test_large_pool_capped_and_deterministic(self):
        chains = [C(str(i % 3), f"t{i}") for i in range(20)]
        v1 = data.diverse8_view("train_en_1", chains); v2 = data.diverse8_view("train_en_1", chains)
        self.assertEqual(len(v1), 8); self.assertEqual(v1, v2)

    @unittest.expectedFailure
    def test_record_view_is_order_invariant(self):
        # KNOWN ISSUE (audited; fixed in q07_audits/fix_pool_view_share.py): permuting the
        # input chains changes which chains the record's view selects.
        chains = [C(str(i % 5), f"t{i}") for i in range(30)]
        a = data.diverse8_view("train_en_1", chains)
        b = data.diverse8_view("train_en_1", list(reversed(chains)))
        self.assertEqual(sorted(c["text"] for c in a), sorted(c["text"] for c in b))

    def test_fixed_view_is_order_invariant(self):
        chains = [C(str(i % 5), f"t{i}") for i in range(30)]
        a = fixed.diverse8_view(chains); b = fixed.diverse8_view(list(reversed(chains)))
        self.assertEqual(sorted(c["text"] for c in a), sorted(c["text"] for c in b))

    @unittest.expectedFailure
    def test_record_view_keeps_true_share(self):
        # KNOWN ISSUE (audited): the record recomputes share from the 8-chain view.
        # 10 x "10" + 10 x "20" has true share 0.5 but the view reports 0.625.
        p = pool(["10"] * 10 + ["20"] * 10)
        data.apply_diverse_views({"train_en_1": p})
        self.assertEqual(p["share"], 0.5)

    def test_fixed_view_keeps_true_share(self):
        p = pool(["10"] * 10 + ["20"] * 10)
        fixed.apply_diverse_views({"train_en_1": p})
        self.assertEqual(p["share"], 0.5); self.assertEqual(len(p["chains"]), 8)


# ------------------------------------------------------------------ the vote
class Initialize(unittest.TestCase):
    def test_thresholds_are_inclusive(self):
        pools = {"train_en_p": pool(["1"] * 7 + ["2"] * 3, lang="en"),      # share exactly 0.70
                 **{f"train_l{i}_p": pool(["1"], lang=f"l{i}") for i in range(10)}}
        st = vote.initialize(pools)
        self.assertEqual(st["train_en_p"], {"answer": "1", "kind": "committed"})

    def test_veto_needs_strong_disagreeing_world(self):
        # 7 of 10 others say "2" (exactly 0.70): strong, disagrees -> vetoed
        pools = {"train_en_p": pool(["1"] * 10, lang="en"),
                 **{f"train_l{i}_p": pool(["2"] if i < 7 else ["1"], lang=f"l{i}") for i in range(10)}}
        self.assertEqual(vote.initialize(pools)["train_en_p"], {"answer": None, "kind": "contested"})
        # 6 of 10: not strong -> committed
        pools[f"train_l6_p"] = pool(["1"], lang="l6")
        self.assertEqual(vote.initialize(pools)["train_en_p"]["kind"], "committed")

    def test_no_other_languages(self):
        pools = {"train_en_p": pool(["1", "1", "1", "2"])}                 # share 0.75, no world
        self.assertEqual(vote.initialize(pools)["train_en_p"]["kind"], "committed")
        pools = {"train_en_p": pool(["1", "2"])}
        self.assertEqual(vote.initialize(pools)["train_en_p"]["kind"], "contested")


class UpdatePool(unittest.TestCase):
    def test_world_decides_local_breaks_ties(self):
        p = pool(["1", "1", "2"])
        self.assertEqual(vote.update_pool(p, ["2"] * 5), "2")            # world wins over local majority
        self.assertEqual(vote.update_pool(p, ["1", "2"]), "1")            # equal world share -> local count
        self.assertEqual(vote.update_pool(p, []), "1")                    # no world -> local plurality

    def test_veto_and_generate_constraint(self):
        p = pool(["1", "1", "2"])
        self.assertIsNone(vote.update_pool(p, ["3"] * 7 + ["1"] * 3))    # strong world answer not in pool
        self.assertEqual(vote.update_pool(p, ["3"] * 6 + ["1"] * 4), "1")  # 60%: not strong

    @unittest.expectedFailure
    def test_veto_membership_uses_match(self):
        # KNOWN ISSUE (audited): `top_world not in counts` is string equality while the
        # scoring uses match(); a world answer within 0.5% but spelled differently vetoes.
        p = pool(["100", "100", "50"])
        self.assertEqual(vote.update_pool(p, ["100.4"] * 10), "100")

    @unittest.expectedFailure
    def test_equal_world_share_tie_is_not_string_order(self):
        # KNOWN ISSUE: final tie-break is max(counts, key=(score, answer)) on STRINGS:
        # with equal world share and equal local counts, "9" beats "10".
        p = pool(["9", "10"])
        self.assertEqual(vote.update_pool(p, ["9", "10"]), "10")


class WeightedDecide(unittest.TestCase):
    def test_regen_dissolves_veto_at_half_weight(self):
        p = pool(["1", "1", "2"])
        others = ["3"] * 8 + ["1"] * 2
        self.assertIsNone(regenerate.weighted_decide(p, others, [], 0.5, False))
        self.assertEqual(regenerate.weighted_decide(p, others, ["3"], 0.5, False), "3")

    def test_oscillator_needs_unanimity(self):
        p = pool(["1", "1", "2"])
        others = ["3"] * 8 + ["1"] * 2
        self.assertIsNone(regenerate.weighted_decide(p, others, ["3", "3", "3", "2"], 0.5, True))
        self.assertEqual(regenerate.weighted_decide(p, others, ["3", "3", "3", "3"], 0.5, True), "3")

    def test_regen_cannot_outvote_pool_alone(self):
        p = pool(["1"] * 4)
        self.assertEqual(regenerate.weighted_decide(p, [], ["2"] * 4, 0.5, False), "1")  # 4 vs 2.0


class Metrics(unittest.TestCase):
    def test_counts(self):
        pools = {"a": pool(["1"], gold="1"), "b": pool(["2"], gold="1"), "c": pool(["1"], gold="1")}
        st = {"a": {"answer": "1"}, "b": {"answer": "2"}, "c": {"answer": None}}
        m = metrics(pools, st, ["b", "c"])
        self.assertEqual(m, {"contested_answered": 1, "contested_accuracy": 0.0,
                             "coverage": round(2 / 3, 4), "precision": 0.5})

    @unittest.expectedFailure
    def test_empty_input(self):
        # KNOWN ISSUE: metrics({}, {}, []) divides by zero.
        self.assertEqual(metrics({}, {}, [])["coverage"], 0.0)


# -------------------------------------------------------------- the checker
class CompareResults(unittest.TestCase):
    def run_tool(self, a, b, *args):
        d = tempfile.mkdtemp()
        Path(d, "a.json").write_text(json.dumps(a)); Path(d, "b.json").write_text(json.dumps(b))
        p = subprocess.run([sys.executable, str(ROOT / "tools" / "compare_results.py"),
                            str(Path(d, "a.json")), str(Path(d, "b.json")), *args],
                           capture_output=True, text=True)
        return p.returncode

    def test_detects_differences(self):
        self.assertEqual(self.run_tool({"x": 1.0, "y": [1, 2]}, {"x": 1.0, "y": [1, 2]}), 0)
        self.assertEqual(self.run_tool({"x": 1.0}, {"x": 1.0001}), 1)
        self.assertEqual(self.run_tool({"x": 1.0}, {"x": 1.0001}, "--tol", "0.001"), 0)
        self.assertEqual(self.run_tool({"x": [1, 2]}, {"x": [1]}), 1)
        self.assertEqual(self.run_tool({"x": 1}, {"y": 1}), 1)

    def test_nan_is_not_equal_to_anything(self):
        # was a bug (fixed 2026-10-01): `abs(a - b) > tol` is False when either side is NaN,
        # so a NaN in a re-run passed as IDENTICAL against any recorded number.
        self.assertEqual(self.run_tool({"x": float("nan")}, {"x": 0.5}), 1)
        self.assertEqual(self.run_tool({"x": float("nan")}, {"x": float("nan")}), 1)

    def test_bool_vs_int(self):
        # was a bug (fixed 2026-10-01): True == 1 in Python, so a flag that changed type passed.
        self.assertEqual(self.run_tool({"x": True}, {"x": 1}), 1)
        self.assertEqual(self.run_tool({"x": True}, {"x": True}), 0)


if __name__ == "__main__":
    unittest.main()
