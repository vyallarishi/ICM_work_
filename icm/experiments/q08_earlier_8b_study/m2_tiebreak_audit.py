# M2 AUDIT - independent re-derivation (no rung5 imports), tolerance matching,
# strength sweep, full-system view.
import glob, json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import pyarrow.parquet as pq

RUN="/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
GEN="/lustre10/scratch/rvyalla/multilingual-icm/results/qwen3-8b-native-6shot-main/final/pooled/generated"
TIER={"en":"high","cs":"high","af":"high","de":"high","vi":"high","zh-tw":"high","tr":"high","uk":"high","el":"high","es":"high","fr":"high","pt":"high","nl":"high","pl":"high","id":"high","ms":"high","hi":"mid","th":"mid","bn":"mid","ru":"mid","ar":"mid","ko":"mid","fa":"mid","it":"mid","bg":"mid","ur":"mid","mr":"mid","kk":"mid","az":"mid","kn":"low","sw":"low","eu":"low","am":"low","ml":"low","is":"low","ky":"low","ceb":"low"}
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

gold={}
for L in TIER:
    for line in open(f"{GEN}/{L}.jsonl",encoding="utf-8"):
        r=json.loads(line); gold[r["id"]]=norm(r.get("gold_number"))
votes=defaultdict(Counter)
for f in glob.glob(f"{RUN}/export/**/broad.parquet",recursive=True):
    for r in pq.read_table(f,columns=["instance_id","canonical_answer","parse_status"]).to_pylist():
        a=norm(r["canonical_answer"])
        if r["parse_status"]=="parsed" and a is not None:
            votes[r["instance_id"]][a]+=1

plur=defaultdict(dict)
for iid,c in votes.items():
    if not c: continue
    L="_".join(iid.split("_")[1:-1]); P=iid.rsplit("_",1)[-1]
    plur[P][L]=max(c,key=lambda a:(c[a],a))

print("=== AUDIT 1: leakage ===")
leak=0; items=[]
for iid,c in votes.items():
    g=gold.get(iid); tot=sum(c.values())
    if g is None or not tot: continue
    top=max(c,key=lambda a:(c[a],a))
    if c[top]/tot >= 0.70: continue
    L="_".join(iid.split("_")[1:-1]); P=iid.rsplit("_",1)[-1]
    others={l:p for l,p in plur[P].items() if l!=L}
    if L in others: leak+=1
    items.append({"iid":iid,"tier":TIER[L],"gold":g,"rung0":top,"pool":c,
                  "maj_counts":Counter(others.values()),"n_others":len(others)})
print(f"split items: {len(items)}; target language inside its own reference: {leak} (must be 0)")

print("\n=== AUDIT 2: exact vs tolerance matching ===")
nx=nt=0
for it in items:
    if not it["maj_counts"]:
        it["m2_exact"]=it["m2_tol"]=it["rung0"]; continue
    maj=max(it["maj_counts"],key=lambda a:(it["maj_counts"][a],a))
    it["m2_exact"]=maj if maj in it["pool"] else it["rung0"]
    hit=next((a for a in it["pool"] if match(a,maj)),None)
    it["m2_tol"]=hit if hit is not None else it["rung0"]
    nx+=it["m2_exact"]!=it["rung0"]; nt+=it["m2_tol"]!=it["rung0"]
for key,n in (("m2_exact",nx),("m2_tol",nt)):
    hits=sum(match(it[key],it["gold"]) for it in items)
    print(f"{key}: accuracy {100*hits/len(items):.1f}% (changed {n})")

print("\n=== AUDIT 3: majority-strength sweep ===")
print(f"{'min-share':>10}{'applied':>9}{'M2-acc':>9}{'rest-acc':>10}{'blended':>9}{'low applied/acc':>18}")
for s in (0.0,0.5,0.6,0.7,0.8,0.9):
    applied=[]; rest=[]
    for it in items:
        done=False
        if it["maj_counts"]:
            maj,mn=max(it["maj_counts"].items(),key=lambda kv:(kv[1],kv[0]))
            if mn/max(it["n_others"],1)>=s and maj in it["pool"]:
                applied.append((it,maj)); done=True
        if not done: rest.append(it)
    ah=sum(match(m,it["gold"]) for it,m in applied)
    rh=sum(match(it["rung0"],it["gold"]) for it in rest)
    la=[(it,m) for it,m in applied if it["tier"]=="low"]
    lh=sum(match(m,it["gold"]) for it,m in la)
    print(f"{s:>10.1f}{len(applied):>9}{100*ah/max(len(applied),1):>8.1f}%"
          f"{100*rh/max(len(rest),1):>9.1f}%{100*(ah+rh)/len(items):>8.1f}%"
          f"{len(la):>10}/{100*lh/max(len(la),1):.1f}%")

print("\n=== AUDIT 4: full system ===")
per=defaultdict(lambda:[0,0]); land_n=land_ok=0
for iid,c in votes.items():
    g=gold.get(iid); tot=sum(c.values())
    if g is None or not tot: continue
    top=max(c,key=lambda a:(c[a],a))
    if c[top]/tot>=0.70:
        land_n+=1; ok=match(top,g); land_ok+=ok
        t=per[TIER["_".join(iid.split("_")[1:-1])]]; t[0]+=1; t[1]+=ok
m2h=0
for it in items:
    ok=match(it["m2_exact"],it["gold"]); m2h+=ok
    t=per[it["tier"]]; t[0]+=1; t[1]+=ok
total=land_n+len(items)
print(f"landslides (Rung 0): {land_n} @ {100*land_ok/land_n:.1f}%")
print(f"split pools (M2):    {len(items)} @ {100*m2h/len(items):.1f}%")
print(f"FULL SYSTEM @ 100% coverage: {100*(land_ok+m2h)/total:.1f}% on {total} items")
for tier in ("high","mid","low"):
    n,ok=per[tier]; print(f"  {tier}: {100*ok/n:.1f}% over {n}")
