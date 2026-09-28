"""Item 2: do stage8/18/19/20/41 share the identical 6,149-chain sample?
Test empirically on cached outputs (no model needed), then structurally.
"""
import json

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

s8  = load("stage8_chains.jsonl")
s18 = load("stage18_chains.jsonl")
s19 = load("stage19_chains.jsonl")
s20 = load("stage20_chains.jsonl")

KEYS = ("instance_id","lang","tier","is_gold","contested")
def sig(rows): return [tuple(r[k] for k in KEYS) for r in rows]

a8, a18, a19 = sig(s8), sig(s18), sig(s19)
print("n:", len(a8), len(a18), len(a19), len(s20))
print("stage8 == stage18 (row-by-row, same order):", a8 == a18)
print("stage8 == stage19 (row-by-row, same order):", a8 == a19)

# stage20 is a subset?
a20 = sig(s20)
print("stage20 rows all appear in stage8 multiset:",
      __import__("collections").Counter(a20) <= __import__("collections").Counter(a8))
# is stage20 order-preserving subsequence of stage8?
it = iter(a8)
print("stage20 is an ordered subsequence of stage8:", all(x in it for x in a20))

# Are the signatures even DISCRIMINATIVE? A pool contributes up to 2 gold +
# 2 non-gold chains with identical (iid,lang,tier,is_gold,contested).
from collections import Counter
c = Counter(a8)
dup = sum(v-1 for v in c.values() if v > 1)
print(f"\nWARNING: signature is non-unique for {dup} rows "
      f"({len(c)} distinct signatures for {len(a8)} rows)")
print("  -> equality above cannot distinguish WHICH chain was drawn,")
print("     only which (pool, gold-status) slots were filled.")
