"""Stage 22 — GENERATIVE MP: world-guided pool expansion after convergence.

THE REFRAME: every judge (stages 8/18/19/20/21) reweighted a FIXED sample and
was redundant wherever discrimination (0.65) < consensus (0.97). This stage
uses MP in the GENERATIVE direction -- sample new target-language chains from
P(chain | world's reasoning) -- changing what voting reads instead of
second-guessing what it says. It attacks the proven dead zones: 78/79 final
abstentions have ZERO correct chains in-pool (voting oracle 0%), and 13% of
answered contested pools converge wrong.

CORRELATED-ERROR CONTAINMENT (designed against judge_forensics.log):
  * LATE SEEDING: regenerate from the CONVERGED w0 state (world precision
    95.5% vs 88.7% at init; cascade flips 14->9->4->0 by lap 4).
  * MINORITY PEER: 2 peers from the world-majority answer cluster + 1 from
    the largest minority cluster -- arbitration, not amplification (the
    problem-107 trap keeps the right answer alive in the minority).
  * FRACTIONAL VOTES: regen chains vote at alpha (arms 0.5 / 1.0); veto and
    withdraw semantics unchanged. Abstention recovery needs no new rule:
    the veto fired because the world answer was absent from the pool; a
    legitimate in-language regeneration of it dissolves the veto by the
    existing logic.
  * OSCILLATOR BAR: pools whose answer changed >=2 times in the w0 replay
    (18/20 ended wrong) require regen unanimity to commit; else stay NULL.
  * Single round; regen chains excluded from demo banks and harvest.

PRE-REGISTERED READS (15 langs, vs converged w0 86.61 contested / 95.53 prec):
  * harm FIRST: correct->wrong flips among answered contested pools ~ 0
    and system precision not below w0's.
  * abstained pools (79, forced-accuracy 0%): gold-in-pool recovery and
    correct commits > 0 = capability voting cannot have.
  * contested accuracy: "pulls its weight" bar = w0 + 4 points.

Gold-blind: peers/demos from model outputs only; gold appears in metrics.
Usage: --self-test | --phase gen [--dry-run] (vLLM venv)
       | --phase decide (CPU) | via stage22_regen.sbatch
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from ..q01_can_voting_label_data.consensus_loop_15lang import initialize, update_pool, world_context, metrics
from ..q02_can_the_model_judge_itself.continuation_consistency_judge import extract_answer
from ..q02_can_the_model_judge_itself.ensemble_judge_in_loop import static_pos_demos

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
REGEN_CACHE = OUT_DIR / "stage22_regen.jsonl"
M_SAMPLES = 4
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 400
PEER_MAX_CHARS = 600
MAX_MODEL_LEN = 4096
ALPHAS = (0.5, 1.0)
LANDSLIDE = 0.70
WORLD_STRENGTH = 0.70
ITERATIONS = 4
SEED = 20260816


def converge_w0(pools):
    """Deterministic w0 replay. Returns final state, contested ids, and the
    oscillator set (answer changed >=2 times; 18/20 end wrong)."""
    state = initialize(pools)
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    history = {i: [state[i]["answer"]] for i in contested}
    for _ in range(ITERATIONS):
        new_state = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            new_state[iid]["answer"] = update_pool(
                pools[iid], world_context(pools, state, iid), None, 0.0)
        state = new_state
        for iid in contested:
            history[iid].append(state[iid]["answer"])
    oscillators = {i for i in contested
                   if sum(a != b for a, b in zip(history[i], history[i][1:]))
                   >= 2}
    return state, contested, oscillators


def answer_clusters(pools, state, problem, target_lang):
    """Other languages' CONVERGED answers for this problem, grouped:
    majority cluster answer, largest minority answer (or None)."""
    votes = []
    for iid, s in state.items():
        if iid.rsplit("_", 1)[-1] != problem:
            continue
        if pools[iid]["lang"] == target_lang or s["answer"] is None:
            continue
        votes.append((pools[iid]["lang"], s["answer"]))
    if not votes:
        return None, None, {}
    counts = Counter(a for _, a in votes)
    ranked = counts.most_common()
    majority = ranked[0][0]
    minority = ranked[1][0] if len(ranked) > 1 else None
    by_answer = defaultdict(list)
    for lang, answer in votes:
        by_answer[answer].append(lang)
    return majority, minority, by_answer


def pick_peer_chain(pools, problem, lang, answer):
    """A chain from lang's pool supporting `answer`, shortest under cap."""
    pool = pools.get(f"train_{lang}_{problem}")
    if pool is None:
        return None
    cands = [c for c in pool["chains"]
             if match(c["answer"], answer) and len(c["text"]) <= PEER_MAX_CHARS]
    return min(cands, key=lambda c: len(c["text"]))["text"] if cands else None


def select_regen_peers(pools, state, iid, rng):
    """2 majority-cluster peers + 1 minority-cluster peer (arbitration)."""
    problem = iid.rsplit("_", 1)[-1]
    lang = pools[iid]["lang"]
    majority, minority, by_answer = answer_clusters(pools, state, problem, lang)
    if majority is None:
        return []
    peers = []
    maj_langs = list(by_answer[majority])
    rng.shuffle(maj_langs)
    for peer_lang in maj_langs:
        text = pick_peer_chain(pools, problem, peer_lang, majority)
        if text:
            peers.append(text)
        if len(peers) == 2:
            break
    if minority is not None:
        min_langs = list(by_answer[minority])
        rng.shuffle(min_langs)
        for peer_lang in min_langs:
            text = pick_peer_chain(pools, problem, peer_lang, minority)
            if text:
                peers.append(text)
                break
    return peers


def build_regen_prompt(question, peers, demos):
    """Target-language format demos first (anchor the output language and
    '#### N' format), then world peers, then the open question."""
    blocks = []
    for demo, label in demos:
        if not label:
            continue
        blocks.append("[Q]\n" + demo["question"].strip() + "\n[A]\n"
                      + demo["chain_text"].strip()[:PEER_MAX_CHARS])
    for peer in peers:
        blocks.append("[S]\n" + peer.strip()[:PEER_MAX_CHARS])
    blocks.append("[Q]\n" + question.strip() + "\n[A]\n")
    return "\n\n".join(blocks)


def weighted_decide(pool, others, regen_answers, alpha, unanimity_required):
    """update_pool semantics over the alpha-weighted expanded pool."""
    counts = Counter()
    for c in pool["chains"]:
        counts[c["answer"]] += 1.0
    regen_counts = Counter(regen_answers)
    for a, n in regen_counts.items():
        counts[a] += alpha * n
    if not counts:
        return None
    if unanimity_required and regen_answers and len(regen_counts) > 1:
        regen_counts = Counter()          # oscillator without unanimity:
        counts = Counter()                # regen may not drive a commit
        for c in pool["chains"]:
            counts[c["answer"]] += 1.0
        if not counts:
            return None
    if others:
        top_world, top_n = Counter(others).most_common(1)[0]
        if top_n / len(others) >= WORLD_STRENGTH and top_world not in counts:
            return None
    def score(a):
        ws = (sum(match(o, a) for o in others) / len(others)) if others else 0.0
        return ws + 0.001 * counts[a]
    return max(counts, key=lambda a: (score(a), a))


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- answer_clusters / peers ---")
    def mk(lang, prob, share, plur, answers, gold="10"):
        return (f"train_{lang}_{prob}",
                {"lang": lang, "gold": gold, "question": "q", "share": share,
                 "plurality": plur,
                 "chains": [{"answer": a, "text": f"{lang} chain {a} {i}"}
                            for i, a in enumerate(answers)]})
    pools = dict([mk("en", 1, 0.8, "10", ["10"] * 4),
                  mk("de", 1, 0.8, "10", ["10"] * 4),
                  mk("fr", 1, 0.8, "20", ["20"] * 4),
                  mk("am", 1, 0.5, "30", ["30", "30", "10", "40"])])
    state = {i: {"answer": p["plurality"], "kind": "contested"}
             for i, p in pools.items()}
    majority, minority, by_ans = answer_clusters(pools, state, "1", "am")
    check("majority is 10", majority, "10")
    check("minority is 20", minority, "20")
    peers = select_regen_peers(pools, state, "train_am_1", random.Random(0))
    check("2 majority + 1 minority peers", len(peers), 3)
    check("minority peer present", any("fr chain 20" in p for p in peers), True)

    print("--- weighted_decide ---")
    pool = {"chains": [{"answer": "30"}, {"answer": "30"}, {"answer": "10"}]}
    check("regen votes tip the local balance",
          weighted_decide(pool, ["10"] * 8 + ["30"] * 2, ["10"] * 4, 1.0,
                          False), "10")
    check("veto dissolves when regen produces the world answer",
          weighted_decide({"chains": [{"answer": "30"}]}, ["99"] * 10,
                          ["99"] * 3, 0.5, False), "99")
    check("veto holds when regen does not produce it",
          weighted_decide({"chains": [{"answer": "30"}]}, ["99"] * 10,
                          ["30"] * 3, 0.5, False), None)
    check("oscillator: split regen may not drive commit",
          weighted_decide({"chains": [{"answer": "30"}]}, ["99"] * 10,
                          ["99", "99", "98"], 1.0, True), None)
    check("oscillator: unanimous regen commits",
          weighted_decide({"chains": [{"answer": "30"}]}, ["99"] * 10,
                          ["99", "99", "99"], 1.0, True), "99")
    check("alpha 1.0: regen outvotes local",
          weighted_decide(pool, [], ["10"] * 3, 1.0, False), "10")
    check("alpha 0.25: local outvotes regen",
          weighted_decide(pool, [], ["10"] * 3, 0.25, False), "30")

    print("--- prompt ---")
    demos = [({"question": "DQ", "chain_text": "DC\n#### 1"}, True)]
    prompt = build_regen_prompt("Q?", ["P1", "P2", "P3"], demos)
    check("demos before peers", prompt.index("DC") < prompt.index("P1"), True)
    check("ends open", prompt.endswith("[Q]\nQ?\n[A]\n"), True)

    print("--- converge_w0 on toy world ---")
    state2, contested, osc = converge_w0(pools)
    check("am contested", "train_am_1" in contested, True)

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def phase_gen(args):
    pools = load_pools(args.root)
    state, contested, oscillators = converge_w0(pools)
    demos = static_pos_demos(pools)
    rng = random.Random(SEED + 22)
    jobs = []
    for iid in contested:
        peers = select_regen_peers(pools, state, iid, rng)
        if not peers:
            continue
        prompt = build_regen_prompt(pools[iid]["question"], peers,
                                    demos.get(pools[iid]["lang"], []))
        jobs.append((iid, prompt))
    abstained = [i for i in contested if state[i]["answer"] is None]
    print(f"pools {len(pools)}; contested {len(contested)} "
          f"(abstained {len(abstained)}, oscillators {len(oscillators)}); "
          f"regen jobs {len(jobs)}")
    if args.dry_run:
        print(f"\n--- sample regen prompt (first 500) ---\n{jobs[0][1][:500]}")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=SEED)
    params = SamplingParams(n=M_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P,
                            max_tokens=MAX_NEW_TOKENS, stop=["[Q]", "[S]"],
                            seed=SEED)
    outputs = llm.generate([p for _, p in jobs], params)
    with REGEN_CACHE.open("w", encoding="utf-8") as handle:
        for (iid, _), out in zip(jobs, outputs):
            answers = [extract_answer(o.text) for o in out.outputs]
            handle.write(json.dumps(
                {"iid": iid, "answers": [a for a in answers if a is not None]},
                ensure_ascii=False) + "\n")
    print(f"wrote {REGEN_CACHE} ({len(jobs)} pools)")


def phase_decide(args):
    pools = load_pools(args.root)
    state, contested, oscillators = converge_w0(pools)
    base = metrics(pools, state, contested)
    print(f"converged w0 baseline: {base}")
    regen = {}
    for line in REGEN_CACHE.open(encoding="utf-8"):
        row = json.loads(line)
        regen[row["iid"]] = row["answers"]
    print(f"regen cache: {len(regen)} pools, "
          f"{sum(len(a) for a in regen.values())} parsed answers")

    results = {"w0_converged": base}
    for alpha in ALPHAS:
        new_state = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            others = world_context(pools, new_state, iid)
            new_state[iid]["answer"] = weighted_decide(
                pools[iid], others, regen.get(iid, []), alpha,
                iid in oscillators)
        row = metrics(pools, new_state, contested)

        answered_before = {i for i in contested
                           if state[i]["answer"] is not None}
        flips_cw = flips_wc = 0
        for iid in answered_before:
            was = match(state[iid]["answer"], pools[iid]["gold"])
            now_ans = new_state[iid]["answer"]
            now = now_ans is not None and match(now_ans, pools[iid]["gold"])
            if was and not now:
                flips_cw += 1
            elif not was and now:
                flips_wc += 1
        abst = [i for i in contested if state[i]["answer"] is None]
        recovered = [i for i in abst if new_state[i]["answer"] is not None]
        rec_ok = sum(match(new_state[i]["answer"], pools[i]["gold"])
                     for i in recovered)
        gold_injected = sum(
            1 for i in abst
            if any(match(a, pools[i]["gold"]) for a in regen.get(i, [])))
        row.update({
            "harm_correct_to_wrong": flips_cw,
            "gain_wrong_to_correct": flips_wc,
            "abstained_before": len(abst),
            "abstained_gold_injected": gold_injected,
            "abstained_recovered": len(recovered),
            "abstained_recovered_correct": rec_ok,
        })
        results[f"alpha_{alpha}"] = row
        print(f"[alpha={alpha}] {row}")
    (OUT_DIR / "stage22_results.json").write_text(json.dumps(results, indent=2))
    print("wrote stage22_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--phase", choices=("gen", "decide"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.phase == "gen":
        phase_gen(args)
    elif args.phase == "decide":
        phase_decide(args)
    else:
        parser.error("need --self-test or --phase {gen,decide}")


if __name__ == "__main__":
    main()
