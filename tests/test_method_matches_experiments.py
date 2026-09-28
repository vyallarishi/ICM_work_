"""Prove that icm/method gives exactly the same output as the experiment code
that produced the headline (experiments/q04_full_scale_headline/full_scale_37x1500.py and
the modules it imports).

Builds a synthetic generation run on disk (15 languages x 60 problems, with
landslides, splits, vetoes, oscillators, truncated chains, unparseable answers
and near-equal numbers), then runs both code paths on it, stage by stage, and
requires identical results. No data, no GPU, ~5 seconds:

    python -m unittest tests/test_method_matches_experiments.py
"""
import copy
import json
import random
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


class _BlockHeavyImports:
    """torch/vllm/transformers must never load on a login node."""
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "vllm", "transformers"):
            raise ImportError(f"blocked heavy import: {name}")


sys.meta_path.insert(0, _BlockHeavyImports())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icm.method import data, regenerate, run, vote                       # noqa: E402
from icm.method.metrics import metrics as method_metrics                 # noqa: E402
from icm.experiments.q02_can_the_model_judge_itself import verdict_judge_gate as s8              # noqa: E402
from icm.experiments.q02_can_the_model_judge_itself import continuation_consistency_judge as s20  # noqa: E402
from icm.experiments.q02_can_the_model_judge_itself import ensemble_judge_in_loop as s21         # noqa: E402
from icm.experiments.q01_can_voting_label_data import consensus_loop_15lang as s9           # noqa: E402
from icm.experiments.q03_does_regeneration_help import regeneration_15lang as s22      # noqa: E402
from icm.experiments.q04_full_scale_headline import full_scale_37x1500 as s25         # noqa: E402

LANGS = ["en", "de", "es", "fr", "ru", "hi", "ar", "ko", "am", "sw", "kn", "eu",
         "ml", "is", "ky"]


def make_root(tmp, seed=7, n_problems=60):
    """Write <tmp>/final/pooled/generated/<lang>.jsonl in the real file format."""
    rng = random.Random(seed)
    gen = Path(tmp) / "final" / "pooled" / "generated"
    gen.mkdir(parents=True)
    golds = [str(rng.randint(5, 500)) for _ in range(n_problems)]
    for lang in LANGS:
        rows = []
        for k in range(n_problems):
            gold = golds[k]
            wrong = [str(int(gold) + d) for d in (1, 7, -3)]   # shared wrong answers
            profile = rng.random()                              # how good this pool is
            cands = []
            for j in range(rng.randint(3, 14)):
                r = rng.random()
                if profile < 0.55:
                    ans = gold if r < 0.85 else rng.choice(wrong)
                elif profile < 0.80:
                    ans = gold if r < 0.50 else rng.choice(wrong)
                else:
                    ans = rng.choice(wrong) if r < 0.80 else gold
                fmt = rng.random()
                pred = ans
                if fmt < 0.10:
                    pred = f"{int(ans):,}"
                elif fmt < 0.15:
                    pred = ans + ".0"
                elif fmt < 0.18:
                    pred = None
                elif fmt < 0.20:
                    pred = "abc"
                text = (f"Step {j}: compute {ans} for problem {k} in {lang}. "
                        * rng.randint(1, 12) + f"\n#### {ans}")
                if rng.random() < 0.03:
                    text = ""
                cands.append({"text": text, "pred_number": pred,
                              "finish_reason": "length" if rng.random() < 0.08 else "stop"})
            if rng.random() < 0.05 and cands:
                cands[0]["pred_number"] = str(round(float(gold) * 1.003, 3))  # near-equal
            rows.append({"id": f"train_{lang}_{k}", "question": f"Question {k} in {lang}?",
                         "gold_number": gold if rng.random() > 0.02 else None,
                         "candidates": cands})
        (gen / f"{lang}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def fake_regen(jobs, pools, seed=11):
    """Pretend the model answered: mostly the world's answer, sometimes not,
    sometimes unanimous, sometimes empty."""
    rng = random.Random(seed)
    regen = {}
    for iid, _ in jobs:
        gold = pools[iid]["gold"]
        wrong = [str(int(float(gold)) + d) for d in (1, 7, -3)]
        r = rng.random()
        if r < 0.15:
            answers = []
        elif r < 0.45:
            answers = [gold] * 4
        elif r < 0.60:
            answers = [rng.choice(wrong)] * 4
        else:
            answers = [gold if rng.random() < 0.6 else rng.choice(wrong) for _ in range(4)]
        regen[iid] = answers
    return regen


class MethodMatchesExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        make_root(cls.root)
        cls.pools_m = data.load(cls.root)
        cls.pools_e = s25.load(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_norm_and_match(self):
        raw = ["1,200", "1200.0", "1200", " 12 ", "abc", None, "1e30", "-0", "0.5",
               "1" + "0" * 30, "NaN", "inf", "3.14159", "100.4", "100", "0", "-7.50", ""]
        for x in raw:
            self.assertEqual(data.norm(x), s8.norm(x), x)
        normed = [data.norm(x) for x in raw]
        for a in normed:
            for b in normed:
                self.assertEqual(data.match(a, b), s8.match(a, b), (a, b))

    def test_load_pools_and_views(self):
        self.assertEqual(self.pools_m, self.pools_e)
        self.assertEqual(data.load_pools(self.root), s8.load_pools(self.root))
        self.assertGreater(len(self.pools_m), 500)

    def test_initialize_and_converge(self):
        init_m, init_e = vote.initialize(self.pools_m), s9.initialize(self.pools_e)
        self.assertEqual(init_m, init_e)
        kinds = [(s["kind"], s["answer"] is None) for s in init_m.values()]
        self.assertIn(("committed", False), kinds)
        self.assertIn(("contested", False), kinds)
        self.assertIn(("contested", True), kinds, "no vetoed pool in synthetic data")

        got_m = vote.converge(self.pools_m)
        got_e = s25.converge(self.pools_e, s25.World(self.pools_e))
        got_w0 = s22.converge_w0(self.pools_e)
        self.assertEqual(got_m, got_e)
        self.assertEqual(got_m, got_w0)
        state, contested, oscillators = got_m
        self.assertTrue(any(state[i]["answer"] is None for i in contested), "no abstentions")
        self.assertTrue(oscillators, "no oscillators in synthetic data")
        self.assertEqual(method_metrics(self.pools_m, state, contested),
                         s9.metrics(self.pools_e, state, contested))

    def test_regeneration_prompts(self):
        state, contested, _ = vote.converge(self.pools_m)
        jobs_m = regenerate.build_jobs(self.pools_m, state, contested)
        demos = s21.static_pos_demos(self.pools_e)
        rng = random.Random(s25.SEED + 25)
        jobs_e = []
        for iid in contested:
            peers = s22.select_regen_peers(self.pools_e, state, iid, rng)
            if not peers:
                continue
            jobs_e.append((iid, s22.build_regen_prompt(
                self.pools_e[iid]["question"], peers,
                demos.get(self.pools_e[iid]["lang"], []))))
        self.assertEqual(jobs_m, jobs_e)
        self.assertGreater(len(jobs_m), 20)
        for text in ["blah\n#### 1,200.", "so 3 + 4 = 7 apples", "no numbers", "#### 12.50\n",
                     "x 1e5", "[S] 42"]:
            self.assertEqual(regenerate.extract_answer(text), s20.extract_answer(text), text)

    def test_full_decision(self):
        state, contested, _ = vote.converge(self.pools_m)
        jobs = regenerate.build_jobs(self.pools_m, state, contested)
        regen = fake_regen(jobs, self.pools_m)
        got_m = run.decide(copy.deepcopy(self.pools_m), regen)

        with tempfile.TemporaryDirectory() as out:
            cache = Path(out) / "stage25_regen.jsonl"
            with cache.open("w", encoding="utf-8") as h:
                for iid, answers in regen.items():
                    h.write(json.dumps({"iid": iid, "answers": answers}) + "\n")
            saved_cache, saved_out = s25.REGEN_CACHE, s25.OUT_DIR
            try:
                s25.REGEN_CACHE, s25.OUT_DIR = cache, Path(out)
                s25.phase_decide(Namespace(root=self.root))
            finally:
                s25.REGEN_CACHE, s25.OUT_DIR = saved_cache, saved_out
            got_e = json.loads((Path(out) / "stage25_results.json").read_text())
            self.assertEqual(regenerate.load_regen(cache), regen)

        self.assertEqual(json.loads(json.dumps(got_m)), got_e)
        self.assertGreater(got_m["with_regen"]["recovered"], 0)
        self.assertGreater(got_m["with_regen"]["abstained_before"],
                           got_m["with_regen"]["recovered"])


if __name__ == "__main__":
    unittest.main()
