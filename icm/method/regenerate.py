"""Step 2 of the method: give a stuck language a second try.

After `vote.converge`, some contested pools abstained: the other languages
strongly agree on an answer this language never produced. We show the language
three other languages' solutions (2 supporting the world's answer, 1 supporting
the largest minority answer) and let it solve the problem again, 4 times.

The new chains vote at half weight (ALPHA = 0.5), under exactly the same rule
as before (`weighted_decide`). If one of them produces the world's answer, the
veto no longer applies and the pool is recovered. A pool that oscillated
during convergence is only allowed to commit if its 4 new chains all agree.

GPU part (`build_jobs` + `generate`): writes one line per pool,
    {"iid": ..., "answers": [...]}
CPU part (`weighted_decide`): re-decides using those answers.
"""
import json
import random
import re
from collections import Counter, defaultdict

from .data import LANDSLIDE, WORLD_STRENGTH, match, norm

M_SAMPLES = 4          # new chains per stuck pool
ALPHA = 0.5            # weight of each new chain relative to an original chain
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 400
PEER_MAX_CHARS = 600
MAX_MODEL_LEN = 4096
SEED = 20260816

ANSWER_RE = re.compile(r"####\s*([^\n]+)")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(continuation):
    """'#### N' if present, else the last number in the continuation."""
    m = ANSWER_RE.search(continuation)
    if m:
        return norm(m.group(1).strip().rstrip("."))
    numbers = NUMBER_RE.findall(continuation)
    return norm(numbers[-1]) if numbers else None


def format_demos(pools):
    """Per language, 2 short chains from landslide pools that agree with the
    pool's own majority. They anchor the output language and the '#### N'
    format. Label-blind: gold is not consulted."""
    demos = defaultdict(list)
    for iid in sorted(pools):
        pool = pools[iid]
        if pool["share"] < LANDSLIDE or len(demos[pool["lang"]]) >= 2:
            continue
        for chain in pool["chains"]:
            if (chain["answer"] == pool["plurality"]
                    and len(chain["text"]) <= 700
                    and len(demos[pool["lang"]]) < 2):
                demos[pool["lang"]].append(
                    ({"question": pool["question"],
                      "chain_text": chain["text"]}, True))
    return demos


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


def build_jobs(pools, state, contested):
    """(iid, prompt) for every contested pool that has peers to learn from."""
    demos = format_demos(pools)
    rng = random.Random(SEED + 25)
    jobs = []
    for iid in contested:
        peers = select_regen_peers(pools, state, iid, rng)
        if not peers:
            continue
        jobs.append((iid, build_regen_prompt(pools[iid]["question"], peers,
                                             demos.get(pools[iid]["lang"], []))))
    return jobs


def generate(jobs, model_dir, out_path):
    """GPU: sample M_SAMPLES chains per prompt with vLLM; write the answers cache."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=SEED)
    params = SamplingParams(n=M_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P,
                            max_tokens=MAX_NEW_TOKENS, stop=["[Q]", "[S]"],
                            seed=SEED)
    outputs = llm.generate([p for _, p in jobs], params)
    with open(out_path, "w", encoding="utf-8") as handle:
        for (iid, _), out in zip(jobs, outputs):
            answers = [extract_answer(o.text) for o in out.outputs]
            handle.write(json.dumps(
                {"iid": iid, "answers": [a for a in answers if a is not None]},
                ensure_ascii=False) + "\n")


def load_regen(path):
    """Read the answers cache written by `generate`: iid -> [answers]."""
    regen = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            regen[row["iid"]] = row["answers"]
    return regen


def weighted_decide(pool, others, regen_answers, alpha, unanimity_required):
    """`vote.update_pool` over the pool plus its new chains at weight alpha."""
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
