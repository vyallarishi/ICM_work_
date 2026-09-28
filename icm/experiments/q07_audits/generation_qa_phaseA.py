import json, glob, statistics as st
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
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
R="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA/final/pooled/generated"
print(f"{'lang':<5}{'items':>6}{'chains':>8}{'trunc%':>8}{'raw acc':>9}{'mix%':>6}{'landslide%':>11}")
rows=[]
for f in sorted(glob.glob(f"{R}/*.jsonl")):
    lang=Path(f).stem; items=0; chains=0; trunc=0; accs=[]; mix=0; land=0
    for line in open(f,encoding="utf-8"):
        r=json.loads(line); g=norm(r.get("gold_number"))
        cands=r.get("candidates",[])
        pool=[]
        for c in cands:
            chains+=1
            if c.get("finish_reason")=="length": trunc+=1; continue
            a=norm(c.get("pred_number"))
            if a is not None: pool.append(a)
        if not pool or g is None: continue
        items+=1
        oks=[match(a,g) for a in pool]
        accs.append(sum(oks)/len(oks))
        if any(oks) and not all(oks): mix+=1
        cnt=Counter(pool)
        if cnt.most_common(1)[0][1]/len(pool)>=0.7: land+=1
    rows.append((lang, items, chains, 100*trunc/max(chains,1), 100*st.mean(accs) if accs else 0, 100*mix/max(items,1), 100*land/max(items,1)))
    print(f"{lang:<5}{items:>6}{chains:>8}{rows[-1][3]:>7.1f}%{rows[-1][4]:>8.1f}%{rows[-1][5]:>5.0f}%{rows[-1][6]:>10.0f}%")
print("DONE")
