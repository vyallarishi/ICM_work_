"""Smoke check for a new generator's 2-shot record: per language parse / empty / truncated / answer-first rates
and chain + plurality accuracy against gold. GATES: parsed >= 90% overall and >= 70% per language; empty <= 5%;
answer-first (< 30 chars before the first '####') <= 10% overall.   python gen_check.py <record root>"""
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1]); tot = Counter(); gates = []
print(f"{'lang':6s} {'pools':>5s} {'parsed':>6s} {'empty':>5s} {'trunc':>5s} {'ans1st':>6s} {'chainAcc':>8s} {'plurAcc':>7s}")
for path in sorted((root / "final/pooled/generated").glob("*.jsonl")):
    c = Counter()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); c["pools"] += 1; cands = r["candidates"]; votes = Counter()
        for x in cands:
            t = x.get("text") or ""; c["chains"] += 1; c["empty"] += not t.strip(); c["parsed"] += x.get("pred_number") is not None
            c["trunc"] += x.get("finish_reason") == "length"; c["right"] += bool(x.get("correct")); c["ans1st"] += "####" in t and len(t.split("####")[0].strip()) < 30
            if x.get("pred_number") is not None: votes[x["pred_number"]] += 1
        if votes: top = votes.most_common(1)[0][0]; c["plur_right"] += any(x["correct"] for x in cands if x.get("pred_number") == top)
    tot += c; pc = lambda k, d="chains": 100 * c[k] / max(1, c[d])
    print(f"{path.stem:6s} {c['pools']:5d} {pc('parsed'):6.1f} {pc('empty'):5.1f} {pc('trunc'):5.1f} {pc('ans1st'):6.1f} {pc('right'):8.1f} {pc('plur_right','pools'):7.1f}")
    if pc("parsed") < 70: gates.append(f"{path.stem}: parsed {pc('parsed'):.0f}%")
pc = lambda k, d="chains": 100 * tot[k] / max(1, tot[d])
print(f"ALL    {tot['pools']:5d} {pc('parsed'):6.1f} {pc('empty'):5.1f} {pc('trunc'):5.1f} {pc('ans1st'):6.1f} {pc('right'):8.1f} {pc('plur_right','pools'):7.1f}")
if pc("parsed") < 90: gates.append(f"overall parsed {pc('parsed'):.1f}%")
if pc("empty") > 5: gates.append(f"overall empty {pc('empty'):.1f}%")
if pc("ans1st") > 10: gates.append(f"overall answer-first {pc('ans1st'):.1f}%")
print("GATES:", "PASS" if not gates else "FAIL -> " + "; ".join(gates))
