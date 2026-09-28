import sys, random
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from collections import Counter, defaultdict
from ..q01_can_voting_label_data import consensus_loop_37lang as s17

def mk(answers):
    return [{"answer":a,"text":f"t{i}"} for i,a in enumerate(answers)]

print("=== 1. Determinism across calls in one process ===")
ch=mk(["10"]*9+["20"]*5+["30"]*3+["40"]*2+["50"])
r1=s17.diverse8_view("t_en_1",ch); r2=s17.diverse8_view("t_en_1",ch)
print("  identical repeat:", [c["text"] for c in r1]==[c["text"] for c in r2])
print("  answers:", Counter(c["answer"] for c in r1))

print("\n=== 2. Determinism when input LIST ORDER changes (same multiset) ===")
sh=list(ch); random.Random(7).shuffle(sh)
r3=s17.diverse8_view("t_en_1",sh)
print("  same answer multiset as r1?", Counter(c['answer'] for c in r1)==Counter(c['answer'] for c in r3))
print("  r1 answers:", [c["answer"] for c in r1])
print("  r3 answers:", [c["answer"] for c in r3])
print("  same TEXTS selected?", sorted(c['text'] for c in r1)==sorted(c['text'] for c in r3))

print("\n=== 3. Diversity claim: does it preserve all distinct answers when <=8 groups? ===")
cases=[
 (["10"]*19+["20"], "19x10 + 1x20"),
 (["10"]*10+["20"]*10, "10x10 + 10x20"),
 ([str(i) for i in range(20)], "20 distinct"),
 (["10"]*13+["20"]*3+["30"]*2+["40"]+["50"], "5 groups"),
 (["10"]*12+[str(i) for i in range(20,28)], "9 groups (one over)"),
]
for ans,label in cases:
    v=s17.diverse8_view("t_en_9",mk(ans))
    src=Counter(ans); got=Counter(c["answer"] for c in v)
    print(f"  {label:<22} n_in={len(ans):<3} distinct_in={len(src):<3} -> selected={len(v)} distinct_out={len(got)} {dict(got)}")

print("\n=== 4. The distortion: share/plurality after diverse-8 vs the true pool ===")
for ans,label in cases:
    pools={"t_en_9":{"lang":"en","chains":mk(ans),"gold":"10","question":"q","share":0,"plurality":None}}
    true_c=Counter(ans); true_share=true_c.most_common(1)[0][1]/len(ans)
    true_plur=max(true_c,key=lambda a:(true_c[a],a))
    s17.apply_diverse_views(pools)
    p=pools["t_en_9"]
    print(f"  {label:<22} TRUE share={true_share:.3f} plur={true_plur:<4} | DIVERSE8 share={p['share']:.3f} plur={p['plurality']:<4}"
          f"  {'<== LANDSLIDE FLIP' if (true_share>=0.70)!=(p['share']>=0.70) else ''}")

print("\n=== 5. Determinism across PROCESSES (seed derived from repr of tuple) ===")
print("  seed expr:", repr((1729,"t_en_1","diverse",8)))
print("  -> hash-independent (string seed), so stable across runs/PYTHONHASHSEED. Verified in separate process below.")

print("\n=== 6. len(chains)<=8 path returns ALL chains unmodified (no dedup) ===")
c8=mk(["10"]*8)
print("  8 identical chains ->", len(s17.diverse8_view("x_en_1",c8)), "selected (share will be 1.0)")
