"""Fir, CPU: arms from a NEW generator's 2-shot record on the core 25 languages (Tier 2, Mistral as generator).
Computes S and A on the record (lib/agreement.py), then writes into <runs>/:
  sft_<tag>_own_unlabelled.jsonl, sft_<tag>_gold_filtered.jsonl, sft_<tag>_s_routed_<student>.jsonl (frozen S
  thresholds from floor_inputs.json), <tag>_agreement.json, <tag>_coverage.json.
    python build_gen_sets.py <record root> <tag> <runs dir> <floor_inputs.json> <runs_core24 dir>"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / "multilingual_icm/ICM_work_"))
sys.path.insert(0, str(Path.home() / "multilingual_icm/data/findings/jobs"))
from agreement import agreement, pluralities
from icm.method import data
from icm.experiments.q09_does_the_dataset_beat_translation.build_sft_sets import completion_of, prompt_of
from icm.experiments.q09_does_the_dataset_beat_translation.build_gold_arm import gold_chain
from build_arms import own_chain
root, tag, runs, inp, core24 = sys.argv[1], sys.argv[2], sys.argv[3], json.load(open(sys.argv[4])), sys.argv[5]
core = inp["core25"]
plur, gold = {}, {}
for path in sorted(Path(root, "final/pooled/generated").glob("*.jsonl")):
    if path.stem in core: plur[path.stem], gold[path.stem] = pluralities(path)
assert set(plur) == set(core), set(core) - set(plur)
A = agreement(plur, core, core)
ag = {l: {"A": A[l], "S": sum(s for _, s in plur[l].values()) / len(plur[l]),
          "plur_acc": sum(data.match(a, gold[l][p]) for p, (a, _) in plur[l].items()) / len(plur[l])} for l in core}
json.dump(ag, open(f"{runs}/{tag}_agreement.json", "w"), indent=1)
trans = {}
for line in open(f"{core24}/sft_translated.jsonl", encoding="utf-8"):
    r = json.loads(line); trans[r["id"]] = r
pools = data.load_pools(root, set(core)); own, gld, cov = {}, {}, defaultdict(Counter)
for iid in sorted(pools):
    p = pools[iid]; L = p["lang"]; cov[L]["pools"] += 1
    c = own_chain(iid, p["chains"])
    if c: own[iid] = {"id": iid, "lang": L, "prompt": prompt_of(p["question"]), "completion": completion_of(c["text"])}; cov[L]["own_kept"] += 1; cov[L]["own_correct"] += data.match(c["answer"], p["gold"])
    g = gold_chain(iid, p["chains"], p["gold"])
    if g: gld[iid] = {"id": iid, "lang": L, "prompt": prompt_of(p["question"]), "completion": completion_of(g["text"])}; cov[L]["gold_kept"] += 1
del pools
def write(name, pick):
    n = Counter()
    with open(f"{runs}/sft_{name}.jsonl", "w", encoding="utf-8") as h:
        for iid in sorted(set(trans) | set(own)):
            if iid.split("_")[1] in core:
                r = pick(iid)
                if r: h.write(json.dumps(r, ensure_ascii=False) + "\n"); n[iid.split("_")[1]] += 1
    print(f"sft_{name}: {sum(n.values())} rows", flush=True); return dict(n)
counts = {"own": write(f"{tag}_own_unlabelled", lambda i: own.get(i)), "gold": write(f"{tag}_gold_filtered", lambda i: gld.get(i))}
routes = {}
for st, t in inp["s_thresholds"].items():
    routes[st] = {l: ("own" if ag[l]["S"] >= t else "translated") for l in core}
    counts[f"s_routed_{st}"] = write(f"{tag}_s_routed_{st}", lambda i, rt=routes[st]: (own if rt[i.split('_')[1]] == "own" else trans).get(i))
json.dump({"tag": tag, "s_thresholds": inp["s_thresholds"], "routes": routes, "counts": counts, "coverage": {l: dict(cov[l]) for l in core}}, open(f"{runs}/{tag}_coverage.json", "w"), indent=1)
for l in core: print(f"{l}: S={ag[l]['S']:.3f} A={ag[l]['A']:.3f} own correct {cov[l]['own_correct']/max(1,cov[l]['own_kept']):.3f}", flush=True)
