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

from .rules import RECORD

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

# Languages added for the 50-language run (pilot 2026-10-01). Tiers proposed, by
# pretraining volume. Kept apart from TIER so every recorded number still means
# "the 37 languages"; pass `languages=LANGUAGES_50` to load_pools to include them.
TIER_NEW = {
    "ja": "high", "zh-cn": "high", "sv": "high",
    "hu": "mid", "he": "mid", "ta": "mid", "te": "mid", "gu": "mid", "ne": "mid", "ka": "mid",
    "my": "low", "yo": "low", "ha": "low",
}
LANGUAGES_50 = {**TIER, **TIER_NEW}

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


def match(a, b, rules=RECORD):
    """Two normalised answers count as equal if they are within 0.5% of each other
    (RECORD), or only if numerically equal (rules.exact_match)."""
    if a is None or b is None:
        return False
    x, y = Decimal(a), Decimal(b)
    if x == y:
        return True
    if rules.exact_match:
        return False
    s = max(abs(x), abs(y))
    return s > 0 and abs(x - y) / s <= Decimal("0.005")


def load_pools(root, languages=None):
    """Read <root>/final/pooled/generated/<lang>.jsonl for every language in
    `languages` (default: TIER, the record's 37).

    Drops chains with no parseable answer and chains cut off by the token limit
    (finish_reason == "length"). Drops pools with no usable chain or no gold.
    """
    languages = TIER if languages is None else languages
    pools = {}
    for path in sorted(glob.glob(f"{root}/final/pooled/generated/*.jsonl")):
        lang = Path(path).stem
        if lang not in languages:
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


def diverse8_view_fixed(chains, k=8):
    """Up to k chains, answer-diverse, deterministic and order-invariant.
    Copied from experiments/q07_audits/fix_pool_view_share.py."""
    if len(chains) <= k:
        return list(chains)
    groups = defaultdict(list)
    for chain in chains:
        groups[chain["answer"]].append(chain)
    # canonical order: biggest answer-group first, ties broken by answer text
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    def canonical(group):
        return sorted(group, key=lambda c: (len(c["text"]), c["text"]))
    selected = [canonical(group)[0] for _, group in ordered[:k]]
    if len(selected) < k:
        chosen = {id(c) for c in selected}
        remainder = [c for _, group in ordered for c in canonical(group)
                     if id(c) not in chosen]
        selected.extend(remainder[:k - len(selected)])
    return selected[:k]


def apply_diverse_views_fixed(pools, k=8):
    """Attach the view WITHOUT touching the pool's true vote statistics.
    Copied from experiments/q07_audits/fix_pool_view_share.py."""
    for pool in pools.values():
        full = pool["chains"]
        counts = Counter(c["answer"] for c in full)
        pool["plurality"] = max(counts, key=lambda a: (counts[a], a))
        pool["share"] = counts[pool["plurality"]] / len(full)
        pool["full_size"] = len(full)
        pool["chains"] = diverse8_view_fixed(full, k)
    return pools


def load(root, rules=RECORD, languages=None):
    """The pools as the headline run saw them (RECORD), or with the fixed view."""
    pools = load_pools(root, languages)
    return apply_diverse_views_fixed(pools) if rules.fixed_view else apply_diverse_views(pools)
