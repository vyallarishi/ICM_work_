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
for lang in ("en","am"):
    for label, root in (("30B", "qwen3-30b-smoke"), ("14B", "qwen3-14b-smoke")):
        path=f"/scratch/rvyalla/multilingual-icm/results/{root}/final/pooled/generated/{lang}.jsonl"
        items=0; chains=0; trunc=0; accs=[]; mix=0; land=0
        for line in open(path,encoding="utf-8"):
            r=json.loads(line); g=norm(r.get("gold_number"))
            pool=[]
            for c in r.get("candidates",[]):
                chains+=1
                if c.get("finish_reason")=="length": trunc+=1; continue
                a=norm(c.get("pred_number"))
                if a is not None: pool.append(a)
            if not pool or g is None: continue
            items+=1
            oks=[match(a,g) for a in pool]
            accs.append(sum(oks)/len(oks))
            if any(oks) and not all(oks): mix+=1
            if Counter(pool).most_common(1)[0][1]/len(pool)>=0.7: land+=1
        print(f"{lang} {label}: items {items}, chains {chains}, trunc {100*trunc/max(chains,1):.1f}%, "
              f"raw acc {100*st.mean(accs) if accs else 0:.1f}%, mix {100*mix/max(items,1):.0f}%, landslide {100*land/max(items,1):.0f}%")
print("DONE")
