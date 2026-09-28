"""Are the ~0.65 channels the SAME signal or DIFFERENT weak signals?
Join J3 (verdict), cc (continuation), peer-residual scores per chain;
measure pairwise Spearman and ensemble AUCs (z-scored sums) on contested
and harvest splits. All cached; no GPU."""
import json, random, statistics as st
from collections import defaultdict
from .verdict_judge_gate import SEED, auc, build_demo_bank, gate_sample, load_pools, match
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
r8  = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl", encoding="utf-8")]
r19 = [json.loads(l) for l in open(f"{RUN_DIR}/stage19_chains.jsonl", encoding="utf-8")]
r20 = {}
for l in open(f"{RUN_DIR}/stage20_chains.jsonl", encoding="utf-8"):
    d = json.loads(l)
    r20.setdefault(d["instance_id"], []).append(d)

rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
assert len(sample) == len(r8) == len(r19)
# stage20 saved only usable rows; rejoin by (instance, order-within-instance)
seen = defaultdict(int)
for s, a, b in zip(sample, r8, r19):
    s["J3"], s["P3blind"] = a.get("J3"), b.get("P3blind")
cursor = defaultdict(int)
for s in sample:
    iid = s["instance_id"]
    rows = r20.get(iid, [])
    # stage20 kept gate order within instance; advance cursor only on usable
    from .continuation_consistency_judge import truncate_chain
    if truncate_chain(s["text"]) is not None:
        k = cursor[iid]
        if k < len(rows):
            assert rows[k]["is_gold"] == s["is_gold"], (iid, k)
            s["cc"] = rows[k].get("cc")
            cursor[iid] += 1

sc = [s for s in sample if all(s.get(k) is not None for k in ("J3","cc","P3blind"))]
print(f"{len(sc)} chains with all three channels")

def zscore_by_lang(key):
    by = defaultdict(list)
    for s in sc: by[s["lang"]].append(s[key])
    mu = {l: st.mean(v) for l, v in by.items()}
    sd = {l: max(st.pstdev(v), 1e-9) for l, v in by.items()}
    for s in sc: s[key+"_z"] = (s[key]-mu[s["lang"]])/sd[s["lang"]]
for k in ("J3","cc","P3blind"): zscore_by_lang(k)

def spear(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); rk=[0]*len(v)
        for p,i in enumerate(order): rk[i]=p
        return rk
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    cov = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    return cov/(sum((a-mx)**2 for a in rx)**0.5*sum((b-my)**2 for b in ry)**0.5)

print("\npairwise Spearman (all chains / contested only):")
import itertools
for a, b in itertools.combinations(("J3","cc","P3blind"), 2):
    all_r = spear([s[a] for s in sc],[s[b] for s in sc])
    c = [s for s in sc if s["contested"]]
    con_r = spear([s[a] for s in c],[s[b] for s in c])
    print(f"  {a:>8} vs {b:<8} {all_r:+.3f} / {con_r:+.3f}")

def report(key_fn, label):
    g=[key_fn(s) for s in sc if s["is_gold"] and s["contested"]]
    r=[key_fn(s) for s in sc if not s["is_gold"] and s["contested"]]
    gen, con = [], []
    for s in sc:
        if s["contested"]: continue
        pool = pools[s["instance_id"]]
        cr = match(pool["plurality"], pool["gold"])
        if cr and s["is_gold"]: gen.append(key_fn(s))
        elif not cr and not s["is_gold"]: con.append(key_fn(s))
    print(f"{label:<26} contested {auc(g,r):.3f}   harvest {auc(gen,con):.3f}")

print()
report(lambda s: s["J3_z"], "J3 alone (z)")
report(lambda s: s["cc_z"], "cc alone (z)")
report(lambda s: s["P3blind_z"], "peer alone (z)")
report(lambda s: s["J3_z"]+s["cc_z"], "J3 + cc")
report(lambda s: s["J3_z"]+s["cc_z"]+s["P3blind_z"], "all three")
report(lambda s: 2*s["cc_z"]+s["J3_z"], "2cc + J3")
print("DONE")
