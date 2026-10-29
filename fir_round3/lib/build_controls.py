"""Fir, CPU: build the Tier-1 control arms on the 38-language floor set from the shipped floor files and
inputs/floor_inputs.json (per-language self-consistency S, A-route, seed-1 oracle routes). No gold is used except
in the oracle arm (which is a test-set oracle by construction). Writes into <runs>/:
  sft_random_mix.jsonl      per example: translated with probability p = the A-route's translated share, else own
  sft_union.jsonl           translated AND own for every pool (2x examples)
  sft_oracle_<student>.jsonl   per language: the source that won at seed 1 for that student
  sft_s_routed_<student>.jsonl per language: own if S >= the student's frozen threshold, else translated
    python build_controls.py <runs dir> <floor_inputs.json>"""
import json, random, sys
from collections import Counter
runs, inp = sys.argv[1], json.load(open(sys.argv[2]))
def load(name):
    d = {}
    for line in open(f"{runs}/sft_{name}.jsonl", encoding="utf-8"):
        r = json.loads(line); d[r["id"]] = r
    return d
T, O = load("translated"), load("own_unlabelled")
ids = sorted(set(T) | set(O)); lang = lambda i: i.split("_")[1]
def write(name, pick, double=False):
    n = Counter()
    with open(f"{runs}/sft_{name}.jsonl", "w", encoding="utf-8") as h:
        for i in ids:
            for r in (pick(i) if double else [pick(i)]):
                if r is not None:
                    h.write(json.dumps(r, ensure_ascii=False) + "\n"); n[lang(i)] += 1
    print(f"sft_{name}: {sum(n.values())} rows, {len(n)} languages", flush=True); return dict(n)
p = inp["random_mix_translated_share"]
counts = {"random_mix": write("random_mix", lambda i: (T if random.Random(f"mix-{i}").random() < p else O).get(i) or T.get(i) or O.get(i)),
          "union": write("union", lambda i: [T.get(i), O.get(i)], double=True)}
for st, route in inp["s_route"].items():
    counts[f"s_routed_{st}"] = write(f"s_routed_{st}", lambda i, rt=route: (O if rt[lang(i)] == "own_unlabelled" else T).get(i))
for st, route in inp["oracle_route"].items():
    if route:
        counts[f"oracle_{st}"] = write(f"oracle_{st}", lambda i, rt=route: (O if rt[lang(i)] == "own_unlabelled" else T).get(i))
json.dump(counts, open(f"{runs}/controls_counts.json", "w"), indent=1)
