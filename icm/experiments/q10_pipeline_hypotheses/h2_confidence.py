"""H2: a label-free confidence, calibrated on other languages, is calibrated on
languages it never saw.

Every labelled pool gets P(label is correct), computed from the STRUCTURE of
agreement only (own chains, other languages' final answers, how the label was
reached). The model is fit with gold from OTHER languages (4-fold language
split, stage 87) and applied once to the held-out fold, so no example's own
language contributes gold to its confidence.

PRE-REGISTERED, on held-out languages, all folds pooled:
    AUC >= 0.90,  ECE <= 0.03 (isotonic calibration, 10 bins),
    precision of the confidence >= 0.95 stratum >= 0.97 in EVERY tier.

Features (no gold): local share and margin of the pool's own top answer,
number of distinct answers, pool size, share of the pool agreeing with the
final label, fraction of other languages whose final answer equals the label,
share of the most common other-language answer, whether the label is that
answer, how many other languages answered, whether the pool was vetoed in the
first pass, whether the label came from regeneration, tier.

Labels are the method's final answers (voting + regeneration, record rules),
i.e. the dataset as it stands. Writes h2_confidence.json (metrics) and
h2_confidence_per_pool.json ({iid: confidence}).

    python -m icm.experiments.q10_pipeline_hypotheses.h2_confidence [--self-test]
"""
import sys
from collections import Counter

import numpy as np

from ...method import data, vote
from ...method.regenerate import load_regen
from ...paths import RUN_DIR
from ..q09_does_the_dataset_beat_translation.build_sft_sets import committed_answers
from .common import ROOT, auc, by_problem, exact, folds, write

FEATURES = ("local_top_share", "local_margin", "n_distinct", "pool_size", "label_local_share",
            "world_agree", "world_top_share", "label_is_world_top", "n_world", "vetoed_first",
            "from_regen", "tier_high", "tier_mid", "tier_low")


def features(pool_full, label, others, first_kind_vetoed, from_regen, tier):
    """No gold anywhere in here."""
    c = Counter(ch["answer"] for ch in pool_full["chains"]); n = sum(c.values())
    ranked = c.most_common()
    top_share = ranked[0][1] / n
    runner = ranked[1][1] / n if len(ranked) > 1 else 0.0
    said = [a for a in others if a is not None]
    wc = Counter(said)
    world_top = wc.most_common(1)[0][0] if said else None
    return {"local_top_share": top_share, "local_margin": top_share - runner, "n_distinct": float(len(c)),
            "pool_size": float(n), "label_local_share": sum(v for a, v in c.items() if data.match(a, label)) / n,
            "world_agree": (sum(data.match(a, label) for a in said) / len(said)) if said else 0.0,
            "world_top_share": (wc.most_common(1)[0][1] / len(said)) if said else 0.0,
            "label_is_world_top": float(world_top is not None and data.match(world_top, label)),
            "n_world": float(len(said)), "vetoed_first": float(first_kind_vetoed),
            "from_regen": float(from_regen),
            "tier_high": float(tier == "high"), "tier_mid": float(tier == "mid"), "tier_low": float(tier == "low")}


def fit_logistic(x, y, l2=1e-3, iters=50):
    """Unweighted L2 logistic regression by Newton steps (calibrated probabilities,
    unlike the class-balanced probe regression). Returns (mu, sd, w, b)."""
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = (x - mu) / sd
    X = np.hstack([z, np.ones((len(z), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def predict(model, x):
    mu, sd, w = model
    X = np.hstack([(x - mu) / sd, np.ones((len(x), 1))])
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


def isotonic_fit(p, y):
    """Pool-adjacent-violators on predictions sorted ascending. Returns (thresholds, values)."""
    order = np.argsort(p); ps, ys = p[order], y[order].astype(float)
    blocks = [[ys[0], 1.0, ps[0]]]
    for v, x in zip(ys[1:], ps[1:]):
        blocks.append([v, 1.0, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n, _ = blocks.pop(); blocks[-1][0] += s; blocks[-1][1] += n; blocks[-1][2] = x
    return np.array([b[2] for b in blocks]), np.array([b[0] / b[1] for b in blocks])


def isotonic_apply(cal, p):
    th, val = cal
    idx = np.searchsorted(th, p, side="left")
    return val[np.clip(idx, 0, len(val) - 1)]


def ece(p, y, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def self_test():
    ok = True
    def check(name, cond):
        nonlocal ok
        if not cond:
            ok = False; print("FAIL", name)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3)); y = (rng.random(4000) < 1 / (1 + np.exp(-(2 * x[:, 0] - 1)))).astype(float)
    m = fit_logistic(x, y); p = predict(m, x)
    check("logistic recovers the slope sign", m[2][0] > 1.0)
    check("logistic is calibrated in-sample", ece(p, y) < 0.03)
    cal = isotonic_fit(p, y)
    check("isotonic is monotone", np.all(np.diff(cal[1]) >= -1e-12))
    check("isotonic output in [0,1]", isotonic_apply(cal, p).min() >= 0 and isotonic_apply(cal, p).max() <= 1)
    check("auc of a perfect score", auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0)
    C = lambda a: {"answer": a, "text": "x"}
    f = features({"chains": [C("4"), C("4"), C("9")]}, "4", ["4", "9", None, "4"], False, False, "low")
    check("features: world agree 2/3", abs(f["world_agree"] - 2 / 3) < 1e-9)
    check("features: label is world top", f["label_is_world_top"] == 1.0)
    check("features: no gold", "gold" not in f)
    print("ALL SELF-TESTS PASSED" if ok else "SELF-TEST FAILED")
    return ok


def main():
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    if not self_test():
        sys.exit("self-test failed")
    full = data.load_pools(ROOT)
    view = data.apply_diverse_views({k: dict(v, chains=list(v["chains"])) for k, v in full.items()})
    first = vote.initialize(view)
    after_vote, _, _ = vote.converge({k: dict(v) for k, v in view.items()})
    final = committed_answers({k: dict(v) for k, v in view.items()}, load_regen(RUN_DIR / "stage25_regen.jsonl"))
    probs = by_problem(full); fold = folds()

    ids, X, y, fo, tiers = [], [], [], [], []
    for iid, label in final.items():
        if label is None:
            continue
        p = full[iid]; lang = p["lang"]
        others = [final[o] for o in probs[iid.rsplit("_", 1)[-1]] if o != iid and full[o]["lang"] != lang]
        f = features(p, label, others, first[iid]["kind"] == "contested" and first[iid]["answer"] is None,
                     after_vote[iid]["answer"] is None, data.TIER[lang])
        ids.append(iid); X.append([f[k] for k in FEATURES]); y.append(float(exact(label, p["gold"])))
        fo.append(fold[lang]); tiers.append(data.TIER[lang])
    X, y, fo, tiers = np.array(X), np.array(y), np.array(fo), np.array(tiers)
    raw, cal = np.zeros(len(y)), np.zeros(len(y))
    for k in sorted(set(fo)):
        tr, te = fo != k, fo == k
        m = fit_logistic(X[tr], y[tr])
        iso = isotonic_fit(predict(m, X[tr]), y[tr])
        raw[te] = predict(m, X[te]); cal[te] = isotonic_apply(iso, raw[te])
        print(f"fold {k}: train {tr.sum()} pools, test {te.sum()} pools ({len(set(np.array(ids)[te]))} ids)", flush=True)

    out = {"n_labelled": int(len(y)), "base_precision": round(float(y.mean()), 4), "slices": {}}
    for sl in ("ALL", "high", "mid", "low"):
        m = np.ones(len(y), bool) if sl == "ALL" else tiers == sl
        yy, pr, pc = y[m], raw[m], cal[m]
        strata = {}
        n_all_tier = m.sum()
        for t in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
            s = pc >= t
            strata[str(t)] = {"share_of_labelled": round(float(s.mean()), 4),
                              "precision": round(float(yy[s].mean()), 4) if s.any() else None}
        out["slices"][sl] = {"n": int(n_all_tier), "precision": round(float(yy.mean()), 4),
                             "auc": round(auc(pc[yy == 1], pc[yy == 0]), 4) if 0 < yy.sum() < len(yy) else None,
                             "auc_raw": round(auc(pr[yy == 1], pr[yy == 0]), 4) if 0 < yy.sum() < len(yy) else None,
                             "ece_isotonic": round(ece(pc, yy), 4), "ece_raw": round(ece(pr, yy), 4),
                             "brier": round(float(((pc - yy) ** 2).mean()), 4), "strata": strata}
        o = out["slices"][sl]
        print(f"{sl:<5} n {o['n']:>6}  precision {o['precision']:.4f}  AUC {o['auc']}  ECE iso {o['ece_isotonic']} "
              f"(raw {o['ece_raw']})  >=0.95: {strata['0.95']}  >=0.98: {strata['0.98']}")
    a = out["slices"]["ALL"]
    holds = (a["auc"] >= 0.90 and a["ece_isotonic"] <= 0.03 and
             all((out["slices"][t]["strata"]["0.95"]["precision"] or 0) >= 0.97 for t in ("high", "mid", "low")))
    out["H2_holds"] = bool(holds)
    print(f"\nH2 {'HOLDS' if holds else 'FAILS'} (bars: AUC >= 0.90, ECE <= 0.03, precision >= 0.97 at conf >= 0.95 in every tier)")
    write("h2_confidence.json", out)
    write("h2_confidence_per_pool.json", {i: round(float(c), 5) for i, c in zip(ids, cal)})


if __name__ == "__main__":
    main()
