"""AUDIT 2/5: metrics() denominators and oscillator change-counting."""
from ..q01_can_voting_label_data.consensus_loop_15lang import metrics

fails=[]
def ck(n,g,w):
    ok=g==w
    if not ok: fails.append(f"{n}: got {g!r} want {w!r}")
    print(f"  {'PASS' if ok else 'FAIL'}  {n}  (got {g!r})")

print("=== 2. metrics() denominators ===")
# 5 pools: 2 committed-never-contested (1 right 1 wrong),
#          3 contested (1 right, 1 wrong, 1 abstained)
pools = {
  "train_en_1": {"gold":"10"}, "train_de_1": {"gold":"10"},
  "train_hi_1": {"gold":"10"}, "train_am_1": {"gold":"10"},
  "train_sw_1": {"gold":"10"},
}
state = {
  "train_en_1": {"answer":"10","kind":"committed"},   # right, not contested
  "train_de_1": {"answer":"99","kind":"committed"},   # WRONG, not contested
  "train_hi_1": {"answer":"10","kind":"contested"},   # right
  "train_am_1": {"answer":"99","kind":"contested"},   # wrong
  "train_sw_1": {"answer":None ,"kind":"contested"},  # abstained
}
contested=[i for i,s in state.items() if s["kind"]=="contested"]
m=metrics(pools,state,contested)
print(" ",m)
ck("2a coverage denominator == ALL pools (4/5)", m["coverage"], 0.8)
ck("2b precision denominator == all committed (3/4)", m["precision"], 0.75)
ck("2c contested_answered counts non-null contested", m["contested_answered"], 2)
ck("2d contested_accuracy over answered contested", m["contested_accuracy"], 0.5)

print("\n  -- n = len(state), NOT len(pools). Probe divergence: --")
# metrics uses n=len(state); if state ever omitted pools, coverage inflates.
state2=dict(state); del state2["train_sw_1"]
m2=metrics(pools,state2,[i for i in contested if i in state2])
print(f"  state missing the abstained pool -> coverage {m2['coverage']} (was {m['coverage']})")
ck("2e dropping an abstained pool from state INFLATES coverage to 1.0",
   m2["coverage"], 1.0)

print("\n=== 5. oscillator change-counting ===")
def osc(h): return sum(a!=b for a,b in zip(h,h[1:]))
cases=[
 (["10","10","10","10","10"], 0, "stable"),
 (["10","20","20","20","20"], 1, "one flip then stable"),
 (["10","20","10","20","10"], 4, "true oscillator"),
 (["10","20","30","30","30"], 2, "two DIFFERENT changes (not an oscillation)"),
 ([None,"10","10","10","10"], 1, "veto lifted once"),
 ([None,"10",None,"10",None], 4, "flapping in/out of abstention"),
 ([None,None,None,None,None], 0, "permanently abstained"),
 (["10",None,None,None,None], 1, "answered then abstained once"),
]
for h,exp,label in cases:
    got=osc(h)
    ck(f"5:{label} history={h} -> {got} changes, flagged={got>=2}", got, exp)

print("\n  -- history length check: 4 iterations -> 5 entries -> 4 adjacent pairs")
print("     ITERATIONS=4 in stage22/stage25; history seeded with initialize() answer.")
print("     So an oscillator needs >=2 of 4 possible changes.")

print("\n  -- None semantics: None != '10' counts as a CHANGE.")
print("     A pool vetoed at init that resolves once and stays = 1 change (not flagged).")
print("     A pool that flaps NULL<->answer twice IS flagged. This is sane.")

print()
if fails:
    print(f"{len(fails)} FAILURES:")
    for f in fails: print("  "+f)
else: print("all probes matched expectation")
