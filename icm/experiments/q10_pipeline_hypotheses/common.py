"""Shared helpers for the pre-registered hypothesis tests (see README.md).

Scoring is ALWAYS exact numeric equality with gold, whatever rules produced the
labels, so different rule sets are compared on the same yardstick.
"""
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from ...method import data
from ...paths import RUN_DIR

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
FOLDS_FILE = Path("/home/rvyalla/multilingual_icm/data/probe_work_/runs/stage87_folds.json")
SEED = 20261002


def exact(a, b):
    return a is not None and b is not None and Decimal(a) == Decimal(b)


def score(pools, answers, ids=None):
    """coverage and precision (exact) of {iid: answer or None} over `ids` (default: all)."""
    ids = list(pools) if ids is None else list(ids)
    answered = [i for i in ids if answers.get(i) is not None]
    right = sum(exact(answers[i], pools[i]["gold"]) for i in answered)
    return {"n": len(ids), "coverage": round(len(answered) / max(len(ids), 1), 4),
            "precision": round(right / max(len(answered), 1), 4), "correct": right}


def by_tier(pools, answers):
    out = {"ALL": score(pools, answers)}
    for tier in ("high", "mid", "low"):
        out[tier] = score(pools, answers, [i for i, p in pools.items() if data.LANGUAGES_50.get(p["lang"]) == tier])
    return out


def folds():
    """{lang: fold} for the 4-fold language split of stage 87."""
    f = json.loads(FOLDS_FILE.read_text())["folds"]
    return {lang: int(k) for k, langs in f.items() for lang in langs}


def auc(pos, neg):
    """Rank AUC with midrank ties."""
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); i = 0
    ss = s[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def write(name, obj):
    path = RUN_DIR / name
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    print(f"wrote {path}")


def by_problem(pools):
    out = defaultdict(list)
    for iid in pools:
        out[iid.rsplit("_", 1)[-1]].append(iid)
    return out


def plurality(chains):
    c = Counter(ch["answer"] for ch in chains)
    return max(c, key=lambda a: (c[a], a)), c
