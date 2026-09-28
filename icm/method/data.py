"""Read the generated chains and compare answers.

A *pool* is every chain (solution attempt) the model wrote for one problem in
one language. Pool ids look like ``train_<lang>_<problem>``.

    pools[iid] = {
        "lang": "sw", "question": "...", "gold": "42",   # gold is read ONLY by metrics.py
        "chains": [{"answer": "42", "text": "..."}, ...],
        "share": 0.75,        # fraction of chains that give the top answer
        "plurality": "42",    # the top answer
    }
"""
import glob
import json
import random
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

# The 37 languages, grouped by how much training data the model has seen.
TIER = {
    "en": "high", "cs": "high", "af": "high", "de": "high", "vi": "high",
    "zh-tw": "high", "tr": "high", "uk": "high", "el": "high", "es": "high",
    "fr": "high", "pt": "high", "nl": "high", "pl": "high", "id": "high",
    "ms": "high",
    "hi": "mid", "th": "mid", "bn": "mid", "ru": "mid", "ar": "mid",
    "ko": "mid", "fa": "mid", "it": "mid", "bg": "mid", "ur": "mid",
    "mr": "mid", "kk": "mid", "az": "mid",
    "kn": "low", "sw": "low", "eu": "low", "am": "low", "ml": "low",
    "is": "low", "ky": "low", "ceb": "low",
}
TIERS = ("high", "mid", "low")

LANDSLIDE = 0.70       # a pool commits on its own when >= 70% of its chains agree
WORLD_STRENGTH = 0.70  # the other languages "strongly agree" when >= 70% share one answer


def norm(x):
    """Canonical string for a number: '2,190' == '2190.0' == '2190'. None if not a number."""
    try:
        d = Decimal(str(x).replace(",", "").replace(" ", ""))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not d.is_finite():
        return None
    d = d.normalize()
    if d == d.to_integral_value():
        try:
            d = d.quantize(Decimal(1))
        except InvalidOperation:
            return None  # absurd magnitude (degenerate digit-run): unusable
    return f"{d:f}"


def match(a, b):
    """Two normalised answers count as equal if they are within 0.5% of each other."""
    if a is None or b is None:
        return False
    x, y = Decimal(a), Decimal(b)
    if x == y:
        return True
    s = max(abs(x), abs(y))
    return s > 0 and abs(x - y) / s <= Decimal("0.005")


def load_pools(root):
    """Read <root>/final/pooled/generated/<lang>.jsonl for every language in TIER.

    Drops chains with no parseable answer and chains cut off by the token limit
    (finish_reason == "length"). Drops pools with no usable chain or no gold.
    """
    pools = {}
    for path in sorted(glob.glob(f"{root}/final/pooled/generated/*.jsonl")):
        lang = Path(path).stem
        if lang not in TIER:
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold = norm(row.get("gold_number"))
                chains = []
                for c in row.get("candidates", []):
                    answer = norm(c.get("pred_number"))
                    text = c.get("text") or ""
                    if answer is None or not text.strip():
                        continue
                    if c.get("finish_reason") == "length":
                        continue
                    chains.append({"answer": answer, "text": text})
                if not chains or gold is None:
                    continue
                counts = Counter(c["answer"] for c in chains)
                top_n = counts.most_common(1)[0][1]
                pools[row["id"]] = {
                    "gold": gold, "lang": lang, "question": row["question"],
                    "chains": chains, "share": top_n / len(chains),
                    "plurality": max(counts, key=lambda a: (counts[a], a)),
                }
    return pools


def diverse8_view(instance_id, chains):
    """Cut a pool to at most 8 chains: one per distinct answer (largest groups
    first), then fillers. Deterministic per pool id."""
    if len(chains) <= 8:
        return list(chains)
    rng = random.Random((1729, instance_id, "diverse", 8).__repr__())
    shuffled = list(chains)
    rng.shuffle(shuffled)
    groups = defaultdict(list)
    for chain in shuffled:
        groups[chain["answer"]].append(chain)
    ordered = sorted(groups.values(), key=lambda g: -len(g))
    selected = [group[0] for group in ordered[:8]]
    if len(selected) < 8:
        chosen = {id(c) for c in selected}
        for chain in shuffled:
            if id(chain) not in chosen:
                selected.append(chain)
                chosen.add(id(chain))
                if len(selected) == 8:
                    break
    return selected[:8]


def apply_diverse_views(pools):
    """Replace every pool's chains by its 8-chain view and recompute share/plurality.

    Known issue: recomputing `share` from the 8-chain view distorts it (the view
    keeps one chain per answer, so the top group shrinks). This is what the
    headline run used; the corrected version is
    experiments/q07_audits/fix_pool_view_share.py, and the measured effect is in
    recorded_results/RESULTS.md.
    """
    for instance_id, pool in pools.items():
        view = diverse8_view(instance_id, pool["chains"])
        pool["chains"] = view
        counts = Counter(c["answer"] for c in view)
        pool["share"] = counts.most_common(1)[0][1] / len(view)
        pool["plurality"] = max(counts, key=lambda a: (counts[a], a))
    return pools


def load(root):
    """The pools exactly as the headline run saw them."""
    return apply_diverse_views(load_pools(root))
