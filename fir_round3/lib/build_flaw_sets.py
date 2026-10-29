"""Rorqual, once (needs the Qwen record): Tier-2 flaw-tolerance sets. Three competence tiers of three core
languages each (by Qwen own-chain correctness: high es fr id ~0.90; mid tr th tl ~0.85; low uz hy te ~0.80).
Pools restricted to those with at least one correct and one wrong usable chain, identical across mixes. For
p in {0, 0.25, 0.5, 0.75, 1.0}: per pool, a seeded uniform draw u; a random wrong chain if u < p, else a random
correct chain (so the sets are nested as p grows). Plus the translated solutions of the same pools and the test
rows of the three languages. Writes <out>/flaw_<tier>_p<p>.jsonl, flaw_<tier>_translated.jsonl,
eval_<tier>.jsonl, flaw_counts.json.    python build_flaw_sets.py <out dir>"""
import json, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/rvyalla/multilingual_icm/ICM_work_")
from icm.method import data
from icm.experiments.q09_does_the_dataset_beat_translation.build_sft_sets import SEED, completion_of, prompt_of
F = Path("/home/rvyalla/multilingual_icm/data/findings"); SNAP = "/scratch/rvyalla/multilingual-icm/results/labels_core25"
TIERS = {"high": ["es", "fr", "id"], "mid": ["tr", "th", "tl"], "low": ["uz", "hy", "te"]}
MIXES = [0.0, 0.25, 0.5, 0.75, 1.0]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
langs = {l for v in TIERS.values() for l in v}
trans = {}
for line in open(F / "runs_core24/sft_translated.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["lang"] in langs: trans[r["id"]] = r
pools = data.load_pools(SNAP, langs); counts = {}
for tier, tl in TIERS.items():
    keep = {}
    for iid, p in pools.items():
        if p["lang"] not in tl: continue
        ok = [c for c in p["chains"] if completion_of(c["text"]) and data.match(c["answer"], p["gold"])]
        bad = [c for c in p["chains"] if completion_of(c["text"]) and not data.match(c["answer"], p["gold"])]
        if ok and bad and iid in trans: keep[iid] = (p, ok, bad)
    n = Counter(p["lang"] for p, _, _ in keep.values()); counts[tier] = {"pools": dict(n), "per_mix": {}}
    for pm in MIXES:
        name = f"flaw_{tier}_p{int(pm * 100):03d}"; wrong = 0
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as h:
            for iid in sorted(keep):
                p, ok, bad = keep[iid]; rng = random.Random(f"{SEED}-{iid}-flaw")
                u = rng.random(); c = rng.choice(bad) if u < pm else rng.choice(ok); wrong += u < pm
                h.write(json.dumps({"id": iid, "lang": p["lang"], "prompt": prompt_of(p["question"]), "completion": completion_of(c["text"])}, ensure_ascii=False) + "\n")
        counts[tier]["per_mix"][name] = {"rows": len(keep), "wrong": wrong}
    with open(out / f"flaw_{tier}_translated.jsonl", "w", encoding="utf-8") as h:
        for iid in sorted(keep): h.write(json.dumps(trans[iid], ensure_ascii=False) + "\n")
    with open(out / f"eval_{tier}.jsonl", "w", encoding="utf-8") as h:
        for line in open(F / "runs_core24/eval_pipeline.jsonl", encoding="utf-8"):
            if json.loads(line)["lang"] in tl: h.write(line)
    print(tier, counts[tier], flush=True)
json.dump(counts, open(out / "flaw_counts.json", "w"), indent=1)
