"""Attribution: is the residual-AUC gain from FORMAT features or merely from
removing per-LANGUAGE margin offsets? Fit (a) language-only, (b) format-only,
(c) both; compare residual AUCs."""
import json, random
from .verdict_judge_gate import SEED, auc, build_demo_bank, gate_sample, load_pools
from .format_confound import features, FEATS  # reuse
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
rows = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl", encoding="utf-8")]
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
for s, r in zip(sample, rows): s["J3"] = r.get("J3")
scored = [s for s in sample if s["J3"] is not None]
langs = sorted({s["lang"] for s in scored})

def ols_resid(X, y):
    k = len(X[0])
    XtX = [[sum(r[i]*r[j] for r in X) + (1e-6 if i==j else 0.0) for j in range(k)] for i in range(k)]
    Xty = [sum(r[i]*yy for r, yy in zip(X, y)) for i in range(k)]
    A = [row[:] + [b] for row, b in zip(XtX, Xty)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(A[r][c])); A[c], A[p] = A[p], A[c]
        for r in range(k):
            if r != c and A[c][c]:
                f = A[r][c]/A[c][c]
                for cc in range(c, k+1): A[r][cc] -= f*A[c][cc]
    beta = [A[i][k]/A[i][i] for i in range(k)]
    pred = [sum(b*v for b, v in zip(beta, r)) for r in X]
    resid = [yy-p for yy, p in zip(y, pred)]
    ybar = sum(y)/len(y); r2 = 1 - sum(e*e for e in resid)/sum((yy-ybar)**2 for yy in y)
    return resid, r2

y = [s["J3"] for s in scored]
def std_feats(s):
    f = features(s["text"]); return [f[k] for k in FEATS]
raw_f = [std_feats(s) for s in scored]
nf = len(FEATS)
mu = [sum(r[j] for r in raw_f)/len(raw_f) for j in range(nf)]
sd = [max((sum((r[j]-mu[j])**2 for r in raw_f)/len(raw_f))**0.5, 1e-9) for j in range(nf)]
fz = [[(r[j]-mu[j])/sd[j] for j in range(nf)] for r in raw_f]
ldum = [[1.0 if s["lang"]==l else 0.0 for l in langs] for s in scored]

models = {
    "language-only": ldum,
    "format-only  ": [f+[1.0] for f in fz],
    "format+lang  ": [f+d for f, d in zip(fz, ldum)],
}
print(f"{'model':<16}{'R^2':>6}{'ALL':>8}{'contested':>11}{'high':>7}{'mid':>7}{'low':>7}")
gi = {n: [i for i,s in enumerate(scored) if p(s) and s["is_gold"]]
      for n,p in (("ALL",lambda s:True),("contested",lambda s:s["contested"]),
                  ("high",lambda s:s["tier"]=="high"),("mid",lambda s:s["tier"]=="mid"),
                  ("low",lambda s:s["tier"]=="low"))}
ri = {n: [i for i,s in enumerate(scored) if p(s) and not s["is_gold"]]
      for n,p in (("ALL",lambda s:True),("contested",lambda s:s["contested"]),
                  ("high",lambda s:s["tier"]=="high"),("mid",lambda s:s["tier"]=="mid"),
                  ("low",lambda s:s["tier"]=="low"))}
def row(name, vals):
    a = {n: auc([vals[i] for i in gi[n]], [vals[i] for i in ri[n]]) for n in gi}
    return f"{a['ALL']:>8.3f}{a['contested']:>11.3f}{a['high']:>7.3f}{a['mid']:>7.3f}{a['low']:>7.3f}"
print(f"{'raw J3':<16}{'':>6}" + row("raw", y))
for name, X in models.items():
    resid, r2 = ols_resid(X, y)
    print(f"{name:<16}{r2:>6.3f}" + row(name, resid))
print("DONE")
