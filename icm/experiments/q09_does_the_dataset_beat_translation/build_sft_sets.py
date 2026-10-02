"""Q09 step 1 (CPU): the two training sets and the evaluation set.

THE QUESTION. Does fine-tuning Qwen3-14B-Base on the method's own labelled
chains beat fine-tuning it on translated gold solutions?

THE TWO ARMS ARE MATCHED ON EVERYTHING EXCEPT THE SOLUTION TEXT:
  * same model, same languages (the record's 37), same problems
  * the SAME (language, problem) pairs: only pairs the method labelled AND for
    which both a method chain and a translated solution exist; every pair
    appears exactly once in each arm, so counts match per language
  * same prompt  "Question: {question}\\nAnswer:"  (question = the dataset's
    translated question, identical in both arms)
  * same completion normalisation for both: the solution text up to and
    including its "#### N" line, nothing after
  * same training recipe and seed (train_lora.py), same evaluation (eval.py)

    translated  completion = the dataset's translated gold solution ("answer")
    method      completion = ONE chain from the pool that agrees with the
                answer the method committed (cross-lingual voting +
                regeneration, record rules), chosen uniformly at random with a
                per-pool seed. NO GOLD: the committed answer is label-free and
                the chain is not checked against gold.
    With --with-fusion, `method` is the FULL pipeline: its label is replaced by
    the problem-disjoint, per-tier vote x probe fusion (probe_work_ q10) where
    the fusion decided differently. Chains are then picked exactly as above.
    own_unlabelled  one of the model's own chains, random, NO labelling (control)
    method_lenmatch the method's label, the chain closest in length to the
                translated solution (control for "more tokens")
    method_full (with --with-probe) the method with its probe channel: in
                HIGH-RESOURCE pools the vote marked contested, the committed
                answer is the probe's (4-fold, never trained on that language;
                probe_work_ q09_probe_labels_for_finetuning). Every other pool
                is identical to `method`, chain for chain. Still no gold.

Gold is read only AFTER both sets are written, to print an audit (how many
method examples are actually correct); it does not influence either set.

EVALUATION SET. The dataset's TEST split (disjoint from train), first N
problems per language by id, all 37 languages.

    python -m icm.experiments.q09_does_the_dataset_beat_translation.build_sft_sets [--self-test]
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from ...method import data, vote
from ...method.regenerate import ALPHA, load_regen, weighted_decide
from ...paths import RUN_DIR

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
DATASET = "/scratch/rvyalla/multilingual-icm/assets/datasets/gsm8k-106"
SEED = 20261002
N_EVAL = 250
ANSWER_LINE = re.compile(r"^\s*####")


def prompt_of(question):
    return f"Question: {question.strip()}\nAnswer:"


def completion_of(text):
    """Solution text up to and including the first '#### N' line; None if there is none."""
    out = []
    for line in text.strip().split("\n"):
        out.append(line.rstrip())
        if ANSWER_LINE.match(line):
            return " " + "\n".join(out).strip()
    return None


def committed_answers(pools, regen):
    """The record method's final answer per pool (identical to icm.method.run.decide)."""
    world = vote.World(pools)
    state, contested, oscillators = vote.converge(pools, world)
    final = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        others = world.context(final, iid)
        final[iid]["answer"] = weighted_decide(pools[iid], others, regen.get(iid, []), ALPHA,
                                               iid in oscillators)
    return {i: s["answer"] for i, s in final.items()}


def pick_method_chain(iid, full_chains, committed):
    """One chain agreeing with the committed answer, uniform over the full pool, seeded per pool."""
    cands = [c for c in full_chains if data.match(c["answer"], committed) and completion_of(c["text"])]
    if not cands:
        return None
    return random.Random(f"{SEED}-{iid}").choice(cands)


def build_pairs(full_pools, committed, translated):
    """{iid: (method_completion, translated_completion)} for pairs present in both arms."""
    pairs, stats = {}, Counter()
    for iid in sorted(full_pools):
        stats["pools"] += 1
        c = committed.get(iid)
        if c is None:
            stats["method_abstained"] += 1
            continue
        chain = pick_method_chain(iid, full_pools[iid]["chains"], c)
        if chain is None:
            stats["no_method_chain_with_answer_line"] += 1     # e.g. commit came only from regeneration
            continue
        t = translated.get(iid)
        tc = completion_of(t["answer"]) if t else None
        if tc is None:
            stats["no_usable_translation"] += 1
            continue
        pairs[iid] = (completion_of(chain["text"]), tc, chain["answer"])
        stats["kept"] += 1
    return pairs, stats


def load_split_rows(split, langs):
    from datasets import load_from_disk
    rows = {}
    for lang in langs:
        d = load_from_disk(f"{DATASET}/{split}/{lang}")
        for r in d:
            rows[r["id"]] = r
    return rows


def self_test():
    ok = True
    def check(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print(f"FAIL {name}: {got!r} vs {want!r}")
    check("completion cut", completion_of("a\nb\n#### 5\nextra"), " a\nb\n#### 5")
    check("no answer line", completion_of("a\nb"), None)
    check("prompt", prompt_of(" q? "), "Question: q?\nAnswer:")
    C = lambda a, t: {"answer": a, "text": t}
    full = {"train_en_1": {"chains": [C("5", "x\n#### 5"), C("6", "y\n#### 6"), C("5", "no line")]},
            "train_en_2": {"chains": [C("7", "z\n#### 7")]},
            "train_en_3": {"chains": [C("8", "only regen")]},
            "train_en_4": {"chains": [C("9", "w\n#### 9")]}}
    committed = {"train_en_1": "5", "train_en_2": None, "train_en_3": "8", "train_en_4": "9"}
    translated = {"train_en_1": {"answer": "t\n#### 5"}, "train_en_4": {"answer": "no line"}}
    pairs, st = build_pairs(full, committed, translated)
    check("kept", sorted(pairs), ["train_en_1"])
    check("method chain", pairs["train_en_1"][0], " x\n#### 5")
    check("translated", pairs["train_en_1"][1], " t\n#### 5")
    check("stats", (st["method_abstained"], st["no_method_chain_with_answer_line"], st["no_usable_translation"]), (1, 1, 1))
    check("deterministic pick", pick_method_chain("p", [C("5", "a\n#### 5"), C("5", "b\n#### 5")], "5"),
          pick_method_chain("p", [C("5", "a\n#### 5"), C("5", "b\n#### 5")], "5"))
    full2 = {"train_en_9": {"chains": [C("5", "a\n#### 5"), C("6", "b\n#### 6")]},
             "train_en_8": {"chains": [C("5", "c\n#### 5")]}}
    pairs2 = {"train_en_9": (" a\n#### 5", " t\n#### 5", "5"), "train_en_8": (" c\n#### 5", " t\n#### 5", "5")}
    arm, st2 = probe_arm(pairs2, full2, {"train_en_9": "6", "train_en_8": "7"})
    check("probe relabels", arm["train_en_9"], (" b\n#### 6", "6"))
    check("no chain for probe answer -> unchanged", arm["train_en_8"], (" c\n#### 5", "5"))
    check("probe stats", (st2["relabelled"], st2["probe_answer_has_no_chain_with_answer_line"]), (1, 1))
    full3 = {"train_en_7": {"chains": [C("5", "a\n#### 5"), C("5", "a much longer chain\nwith steps\n#### 5"), C("6", "b\n#### 6")]}}
    lm = lenmatch_arm({"train_en_7": (" a\n#### 5", " a much longer chain\nwith steps!\n#### 5", "5")}, full3, {"train_en_7": "5"})
    own = own_unlabelled_arm({"train_en_7": None}, full3)
    check("own_unlabelled is deterministic", own, own_unlabelled_arm({"train_en_7": None}, full3))
    check("own_unlabelled can pick any answer (no label)", own["train_en_7"][1] in ("5", "6"), True)
    check("lenmatch picks the closest length, same answer", lm["train_en_7"], (" a much longer chain\nwith steps\n#### 5", "5"))
    print("ALL SELF-TESTS PASSED" if ok else "SELF-TEST FAILED")
    return ok


PROBE_COMMITS = Path("/home/rvyalla/multilingual_icm/data/probe_work_/runs/q09_probe_commits.json")
FUSION_COMMITS = Path("/home/rvyalla/multilingual_icm/data/probe_work_/runs/q10_fusion_commits_alltiers.json")


def own_unlabelled_arm(pairs, full):
    """own_unlabelled: ONE of the model's own chains for the pool, chosen at random
    with no labelling at all (any answer, as long as it has an answer line). The
    self-distillation control: if it matches `method`, the labelling adds nothing."""
    out = {}
    for iid in pairs:
        cands = [c for c in full[iid]["chains"] if completion_of(c["text"])]
        c = random.Random(f"{SEED}-{iid}-own").choice(cands)
        out[iid] = (completion_of(c["text"]), c["answer"])
    return out


def lenmatch_arm(pairs, full, committed):
    """method_lenmatch: same label as `method`, but the chain whose completion length is
    closest to the translated solution's (ties: the per-pool seeded random order).
    Label-free; the translation is used only as a length reference. Controls for
    "the method arm just has more tokens"."""
    out = {}
    for iid, (mc, tc, ans) in pairs.items():
        cands = [c for c in full[iid]["chains"] if data.match(c["answer"], committed[iid]) and completion_of(c["text"])]
        random.Random(f"{SEED}-{iid}-len").shuffle(cands)
        best = min(cands, key=lambda c: abs(len(completion_of(c["text"])) - len(tc)))
        out[iid] = (completion_of(best["text"]), best["answer"])
    return out


def probe_arm(pairs, full, probe_commits):
    """method_full completions: the probe's answer where it relabelled a pool,
    otherwise exactly the `method` completion."""
    out, stats = {}, Counter()
    for iid, (mc, tc, ans) in pairs.items():
        p = probe_commits.get(iid)
        if p is None:
            out[iid] = (mc, ans)
            continue
        stats["routed"] += 1
        if data.match(p, ans):
            out[iid] = (mc, ans)
            stats["probe_agrees_with_vote"] += 1
            continue
        chain = pick_method_chain(iid, full[iid]["chains"], p)
        if chain is None:
            out[iid] = (mc, ans)
            stats["probe_answer_has_no_chain_with_answer_line"] += 1
            continue
        out[iid] = (completion_of(chain["text"]), chain["answer"])
        stats["relabelled"] += 1
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--with-probe", action="store_true",
                    help=f"also build method_full from {PROBE_COMMITS} (naive contested gate)")
    ap.add_argument("--with-fusion", action="store_true",
                    help=f"also build method_fusion from {FUSION_COMMITS} (per-tier tuned fusion, q10)")
    ap.add_argument("--with-lenmatch", action="store_true", help="also build method_lenmatch (length control)")
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not self_test():
        sys.exit("self-test failed; refusing to build")

    full = data.load_pools(ROOT)
    view = data.apply_diverse_views({k: dict(v, chains=list(v["chains"])) for k, v in full.items()})
    committed = committed_answers(view, load_regen(RUN_DIR / "stage25_regen.jsonl"))
    langs = sorted({p["lang"] for p in full.values()})
    train = load_split_rows("train", langs)
    pairs, stats = build_pairs(full, committed, train)

    out = {}

    def write_arm(name, arm):
        out[name] = RUN_DIR / f"q09_sft_{name}.jsonl"
        with open(out[name], "w", encoding="utf-8") as hf:
            for iid in pairs:
                base = {"id": iid, "lang": full[iid]["lang"], "prompt": prompt_of(train[iid]["question"])}
                hf.write(json.dumps({**base, "completion": arm[iid][0]}, ensure_ascii=False) + "\n")

    vote_arm = {iid: (mc, ans) for iid, (mc, tc, ans) in pairs.items()}
    write_arm("translated", {iid: (tc, None) for iid, (mc, tc, ans) in pairs.items()})
    extra = {}
    if a.with_fusion:
        commits = json.loads(FUSION_COMMITS.read_text())["commits"]
        method_arm, fstats = probe_arm(pairs, full, commits)
        print("fusion channel:", json.dumps(dict(fstats)),
              f"(fusion answers for {len(commits)} pools; {sum(i in pairs for i in commits)} in the matched set)")
    else:
        method_arm = vote_arm
    write_arm("method", method_arm); extra["method"] = method_arm
    label_of = {iid: v[1] for iid, v in method_arm.items()}
    own = own_unlabelled_arm(pairs, full)
    write_arm("own_unlabelled", own); extra["own_unlabelled"] = own
    if a.with_lenmatch:
        lm = lenmatch_arm(pairs, full, label_of)
        write_arm("method_lenmatch", lm); extra["method_lenmatch"] = lm
    full_arm = None
    if a.with_probe:
        commits = json.loads(PROBE_COMMITS.read_text())["commits"]
        full_arm, pstats = probe_arm(pairs, full, commits)
        write_arm("method_full", full_arm)
        print("naive probe channel (contested gate):", json.dumps(dict(pstats)))

    # evaluation set: test split, first N problems per language by numeric id
    test = load_split_rows("test", langs)
    by_lang = defaultdict(list)
    for r in test.values():
        by_lang[r["language"]].append(r)
    with open(RUN_DIR / "q09_eval.jsonl", "w", encoding="utf-8") as h:
        for lang in langs:
            for r in sorted(by_lang[lang], key=lambda r: int(r["original_id"]))[:a.n_eval]:
                h.write(json.dumps({"id": r["id"], "lang": lang, "tier": data.TIER[lang],
                                    "prompt": prompt_of(r["question"]), "gold": r["answer_number"]},
                                   ensure_ascii=False) + "\n")

    # ---- report: equalisation, then the gold audit (after the sets are written)
    per_lang = Counter(full[i]["lang"] for i in pairs)
    print(json.dumps(dict(stats), indent=1))
    print(f"\nexamples per arm: {len(pairs)} (identical pairs in both arms); per language "
          f"min {min(per_lang.values())} max {max(per_lang.values())}")
    print(f"{'lang':<7}{'n':>6}{'method chars':>14}{'translated chars':>18}{'ratio':>8}")
    flagged = []
    for lang in langs:
        ids = [i for i in pairs if full[i]["lang"] == lang]
        m = sum(len(pairs[i][0]) for i in ids) / len(ids); t = sum(len(pairs[i][1]) for i in ids) / len(ids)
        print(f"{lang:<7}{len(ids):>6}{m:>14.0f}{t:>18.0f}{m / t:>8.2f}")
        if not 0.75 <= m / t <= 1.33:
            flagged.append(lang)
    tot_m = sum(len(v[0]) for v in pairs.values()); tot_t = sum(len(v[1]) for v in pairs.values())
    print(f"ALL    total chars method {tot_m}  translated {tot_t}  ratio {tot_m / tot_t:.2f}"
          + (f"   LENGTH-IMBALANCED languages: {flagged}" if flagged else ""))
    for name, arm in extra.items():
        tot = sum(len(v[0]) for v in arm.values())
        print(f"       total chars {name} {tot}  ratio to translated {tot / tot_t:.2f}")
    gold = {i: Decimal(data.norm(full[i]["gold"])) for i in pairs}
    correct = sum(Decimal(v[2]) == gold[i] for i, v in pairs.items())
    print(f"\nAUDIT (gold read here only, after writing): correct answers in the training set")
    print(f"  vote-only label {correct}/{len(pairs)} = {100 * correct / len(pairs):.2f}%")
    for name, arm in extra.items():
        ca = sum(Decimal(v[1]) == gold[i] for i, v in arm.items())
        changed = [i for i in pairs if Decimal(arm[i][1]) != Decimal(pairs[i][2])]
        fixed = sum(Decimal(arm[i][1]) == gold[i] for i in changed)
        broke = sum(Decimal(pairs[i][2]) == gold[i] for i in changed)
        print(f"  {name:<16}{ca}/{len(pairs)} = {100 * ca / len(pairs):.2f}%   "
              f"(answers differing from the vote-only label: {len(changed)}; {fixed} correct where the vote was not, "
              f"{broke} wrong where the vote was right)")
    if full_arm:
        cf = sum(Decimal(v[1]) == gold[i] for i, v in full_arm.items())
        changed = [i for i in pairs if Decimal(full_arm[i][1]) != Decimal(pairs[i][2])]
        fixed = sum(Decimal(full_arm[i][1]) == gold[i] for i in changed)
        broke = sum(Decimal(pairs[i][2]) == gold[i] for i in changed)
        print(f"  method_full  {cf}/{len(pairs)} = {100 * cf / len(pairs):.2f}%   "
              f"(probe changed {len(changed)} labels: {fixed} now correct, {broke} were correct before)")
    print(f"wrote {', '.join(str(v) for v in out.values())}, {RUN_DIR / 'q09_eval.jsonl'} "
          f"({sum(min(len(by_lang[l]), a.n_eval) for l in langs)} eval prompts)")


if __name__ == "__main__":
    main()
