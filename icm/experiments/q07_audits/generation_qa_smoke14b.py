import json, statistics as st
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
def norm(x):
    try:
        d=Decimal(str(x).replace(",","").replace(" ",""))
        if not d.is_finite(): return None
        d=d.normalize()
        if d==d.to_integral_value(): d=d.quantize(Decimal(1))
        return f"{d:f}"
    except (InvalidOperation,TypeError,ValueError): return None
def match(a,b):
    if a is None or b is None: return False
    x,y=Decimal(a),Decimal(b)
    if x==y: return True
    s=max(abs(x),abs(y)); return s>0 and abs(x-y)/s<=Decimal("0.005")
R="/scratch/rvyalla/multilingual-icm/results/qwen3-30b-smoke/final/pooled/generated"
R8="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-smoke/final/pooled/generated"
for lang in ("en","am"):
    for label, path, cap in (("14B", f"{R}/{lang}.jsonl", None), ("8B ", f"{R8}/{lang}.jsonl", 150)):
        pools=defaultdict(list); trunc=0; total=0; lens=[]
        seen_items=set()
        for line in open(path,encoding="utf-8"):
            r=json.loads(line)
            iid=r["id"]
            if cap:
                seen_items.add(iid)
                if len(seen_items)>cap and iid not in list(seen_items)[:cap]: continue
            g=norm(r.get("gold_number"))
            for c in r.get("candidates",[]) if "candidates" in r else [r]:
                pass
            # rows are per-candidate or per-item? handle both:
            cands = r.get("candidates")
            rows = cands if cands else [r]
            for c in rows:
                total+=1
                if c.get("finish_reason")=="length" or c.get("truncated"): trunc+=1
                a=norm(c.get("pred_number") if "pred_number" in c else c.get("answer_number"))
                text=c.get("text") or c.get("chain_text") or ""
                lens.append(len(text))
                if a is not None: pools[iid].append((a,g))
        items=len(pools); chains=total
        acc=[]; mix=0
        for iid,vals in pools.items():
            g=vals[0][1]
            oks=[match(a,g) for a,_ in vals]
            acc.append(sum(oks)/len(oks))
            if any(oks) and not all(oks): mix+=1
        print(f"{lang} {label}: items {items}, chains {chains}, parse-usable {sum(len(v) for v in pools.values())}, "
              f"trunc {100*trunc/max(total,1):.1f}%, raw acc {100*st.mean(acc) if acc else 0:.1f}%, "
              f"mix {100*mix/max(items,1):.0f}%, med chain {int(st.median(lens)) if lens else 0} ch")
print("DONE")
