"""Format confound in the J3 judge margin (CPU, cached margins).

QUESTION: how much of the judge's margin is explained by surface FORM
(LaTeX, bold, numbered steps, headers, length) rather than content?

PRE-REGISTERED READS:
  * R^2 of format features -> margin: how much variance is form.
  * AUC(gold vs rest) of the format-only prediction: if format alone
    discriminates, part of the judge's measured "skill" is a style proxy.
  * AUC of the residual (margin minus format prediction): the judge's
    form-free discrimination. Residual > raw on contested pools means the
    confound was MASKING signal (canonicalization on GPU should then help);
    residual < raw means form was carrying part of the signal.

Method: recompute stage8's gate sample deterministically (validated by
quality_sample.py: 6149 == 6149), join saved J3 margins by order, OLS of
margin on standardized format features + language fixed effects.
"""
import json
import random
import re
from collections import defaultdict
from .verdict_judge_gate import (SEED, auc, build_demo_bank, gate_sample,
                               load_pools)
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
rows = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl",
                                    encoding="utf-8")]
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
assert len(sample) == len(rows), f"sample mismatch {len(sample)} vs {len(rows)}"
for s, r in zip(sample, rows):
    assert s["instance_id"] == r["instance_id"] and s["is_gold"] == r["is_gold"]
    s["J3"] = r.get("J3")
scored = [s for s in sample if s["J3"] is not None]
print(f"joined {len(scored)} chains with J3 margins")

def features(text):
    lines = [l for l in text.split("\n") if l.strip()]
    return {
        "len_chars": len(text) / 1000.0,
        "n_lines": len(lines) / 10.0,
        "latex": float(bool(re.search(r"\\\[|\\\(|\\frac|\\text|\\times|\$", text))),
        "bold": text.count("**") / 2.0,
        "headers": sum(1 for l in lines if re.match(r"\s*#{1,4}\s", l)),
        "numbered": sum(1 for l in lines if re.match(r"\s*\d+[\.\)]\s", l)),
        "calc_annot": text.count("<<") / 5.0,
        "digit_density": sum(c.isdigit() for c in text) / max(len(text), 1),
    }

FEATS = sorted(features("x"))
langs = sorted({s["lang"] for s in scored})
def design_row(s):
    f = features(s["text"])
    return [f[k] for k in FEATS] + [1.0 if s["lang"] == l else 0.0 for l in langs]

X = [design_row(s) for s in scored]
y = [s["J3"] for s in scored]
k = len(X[0])
# standardize the feature columns (not the language dummies)
nf = len(FEATS)
means = [sum(r[j] for r in X) / len(X) for j in range(nf)]
sds = [max((sum((r[j] - means[j]) ** 2 for r in X) / len(X)) ** 0.5, 1e-9)
       for j in range(nf)]
for r in X:
    for j in range(nf):
        r[j] = (r[j] - means[j]) / sds[j]

# OLS via normal equations + Gaussian elimination (k ~ 23)
XtX = [[sum(r[i] * r[j] for r in X) + (1e-6 if i == j else 0.0)
        for j in range(k)] for i in range(k)]
Xty = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(k)]
A = [row[:] + [b] for row, b in zip(XtX, Xty)]
for col in range(k):
    piv = max(range(col, k), key=lambda r: abs(A[r][col]))
    A[col], A[piv] = A[piv], A[col]
    for r in range(k):
        if r != col and A[col][col]:
            f = A[r][col] / A[col][col]
            for c in range(col, k + 1):
                A[r][c] -= f * A[col][c]
beta = [A[i][k] / A[i][i] for i in range(k)]

pred = [sum(b * v for b, v in zip(beta, r)) for r in X]
resid = [yy - p for yy, p in zip(y, pred)]
ybar = sum(y) / len(y)
ss_tot = sum((yy - ybar) ** 2 for yy in y)
ss_res = sum(e * e for e in resid)
print(f"\nformat+lang -> margin R^2 = {1 - ss_res / ss_tot:.3f}")
print("feature coefficients (per SD):")
for name, b in zip(FEATS, beta[:nf]):
    print(f"  {name:<14} {b:+.3f}")

def slice_aucs(label, pred_fn):
    print(f"\n--- {label} ---")
    print(f"{'slice':<12}{'raw J3':>8}{'format-only':>12}{'residual':>10}")
    for name, cond in (("ALL", lambda s: True),
                       ("contested", lambda s: s["contested"]),
                       ("landslide", lambda s: not s["contested"]),
                       ("high", lambda s: s["tier"] == "high"),
                       ("mid", lambda s: s["tier"] == "mid"),
                       ("low", lambda s: s["tier"] == "low")):
        idx = [i for i, s in enumerate(scored) if cond(s)]
        g = [i for i in idx if scored[i]["is_gold"]]
        r = [i for i in idx if not scored[i]["is_gold"]]
        raw = auc([y[i] for i in g], [y[i] for i in r])
        fmt = auc([pred[i] for i in g], [pred[i] for i in r])
        res = auc([resid[i] for i in g], [resid[i] for i in r])
        print(f"{name:<12}{raw:>8.3f}{fmt:>12.3f}{res:>10.3f}")

slice_aucs("AUC gold vs rest", None)
print("\nDONE")
