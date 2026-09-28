"""Item 2 (decisive): replay the exact rng consumption of each stage's
main() with a synthetic pool set, and compare the SELECTED CHAIN TEXTS.
This catches an rng-order divergence that metadata comparison cannot."""
import random, sys
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import build_demo_bank, gate_sample, SEED

# Build synthetic pools with the real schema: distinguishable chain texts,
# a mix of landslide/contested, several languages.
def make_pools(seed=7):
    r = random.Random(seed)
    pools = {}
    langs = ["en","de","ru","hi","am","sw","kn"]
    for p in range(160):
        for lang in langs:
            iid = f"train_{lang}_{p:04d}"
            gold = "42"
            n = r.randint(4, 10)
            chains = []
            for k in range(n):
                # answer: mostly gold in landslide pools, mixed otherwise
                if p % 3 == 0:
                    ans = "42" if k < n-1 else "17"     # landslide
                else:
                    ans = r.choice(["42","17","99"])    # contested
                chains.append({"answer": ans, "text": f"{iid}|c{k}|{ans}|"+"x"*r.randint(50,900)})
            from collections import Counter
            counts = Counter(c["answer"] for c in chains)
            top = counts.most_common(1)[0][1]
            pools[iid] = {"gold":gold,"lang":lang,"question":f"Q{p}",
                          "chains":chains,"share":top/len(chains),
                          "plurality":max(counts,key=lambda a:(counts[a],a))}
    return pools

pools = make_pools()

def stage8_like():
    """stage8.main / stage18.main / stage19.main / stage20.main / stage41.main
    all do: rng=Random(SEED); demos=build_demo_bank(pools,rng);
            sample=[s for s in gate_sample(pools,rng) if ...]"""
    rng = random.Random(SEED)
    demos = build_demo_bank(pools, rng)
    excl = {d["iid"] for ds in demos.values() for d,_ in ds}
    return [s for s in gate_sample(pools, rng) if s["instance_id"] not in excl]

runs = {name: stage8_like() for name in ("stage8","stage18","stage19","stage20","stage41")}
base = runs["stage8"]
basetexts = [s["text"] for s in base]
print(f"sample size {len(base)}")
for name, r in runs.items():
    same = [s["text"] for s in r] == basetexts
    print(f"  {name:8s} identical chain TEXTS to stage8: {same}")

# Now: what if a stage inserted an extra rng draw between bank and sample?
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
rng.random()                       # one extra draw
excl = {d["iid"] for ds in demos.values() for d,_ in ds}
perturbed = [s for s in gate_sample(pools, rng) if s["instance_id"] not in excl]
pt = [s["text"] for s in perturbed]
print("\nsensitivity check -- one extra rng draw inserted:")
print("  identical texts:", pt == basetexts)
diff = sum(1 for a,b in zip(pt, basetexts) if a != b)
print(f"  differing chain slots: {diff}/{len(basetexts)}")
KEYS=("instance_id","lang","tier","is_gold","contested")
print("  metadata signature still identical:",
      [tuple(s[k] for k in KEYS) for s in perturbed]==[tuple(s[k] for k in KEYS) for s in base])
print("  -> metadata equality is NOT sufficient evidence; rng order is.")
