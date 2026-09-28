"""Why did LR2 fail? Distinguish no-signal from artifact using saved scores.
Checks: (1) does chain LENGTH dominate LR2? (2) is either side (LR_pos,
LR_neg) informative alone? (3) length-normalized LR (per-char), (4) rank
correlation LR2 vs J3 margins (are they even measuring related things?)"""
import json, random, statistics as st
from .verdict_judge_gate import SEED, auc, build_demo_bank, gate_sample, load_pools
from ...paths import RUN_DIR

rows18 = [json.loads(l) for l in open(f"{RUN_DIR}/stage18_chains.jsonl", encoding="utf-8")]
rows8 = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl", encoding="utf-8")]
pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
assert len(sample) == len(rows18) == len(rows8)
for s, r18, r8 in zip(sample, rows18, rows8):
    s["LR_pos"], s["LR_neg"] = r18.get("LR_pos"), r18.get("LR_neg")
    s["LR2"], s["J3"] = r18.get("LR2"), r8.get("J3")
sc = [s for s in sample if s["LR2"] is not None]
print(f"{len(sc)} chains with LR2")

def a(vals_fn, label):
    g = [vals_fn(s) for s in sc if s["is_gold"]]
    r = [vals_fn(s) for s in sc if not s["is_gold"]]
    gc = [vals_fn(s) for s in sc if s["is_gold"] and s["contested"]]
    rc = [vals_fn(s) for s in sc if not s["is_gold"] and s["contested"]]
    print(f"{label:<28} ALL {auc(g, r):.3f}   contested {auc(gc, rc):.3f}")

a(lambda s: s["LR2"], "LR2 = pos - neg")
a(lambda s: s["LR_pos"], "LR_pos alone")
a(lambda s: s["LR_neg"], "LR_neg alone (sign as-is)")
a(lambda s: -s["LR_neg"], "-LR_neg")
a(lambda s: len(s["text"]), "chain length (control)")
a(lambda s: s["LR_pos"] / max(len(s["text"]), 1), "LR_pos per char")
a(lambda s: s["LR2"] / max(len(s["text"]), 1), "LR2 per char")

# spearman-ish: rank correlation LR2 vs J3 (both present)
both = [(s["LR2"], s["J3"]) for s in sc if s["J3"] is not None]
def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0]*len(v)
    for pos, i in enumerate(order): rk[i] = pos
    return rk
rx, ry = ranks([b[0] for b in both]), ranks([b[1] for b in both])
mx, my = st.mean(rx), st.mean(ry)
cov = sum((x-mx)*(y-my) for x, y in zip(rx, ry))
rho = cov / (sum((x-mx)**2 for x in rx)**0.5 * sum((y-my)**2 for y in ry)**0.5)
print(f"\nSpearman(LR2, J3) = {rho:.3f}  (n={len(both)})")
# magnitude sanity: LR2 distribution
lr = [s["LR2"] for s in sc]
print(f"LR2 median {st.median(lr):.1f}, IQR {sorted(lr)[len(lr)//4]:.1f}..{sorted(lr)[3*len(lr)//4]:.1f}")
lp = [s["LR_pos"] for s in sc]
print(f"LR_pos median {st.median(lp):.1f} (sum logprob of chain under pos ctx)")
print("DONE")
