import sys
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q01_can_voting_label_data.consensus_loop_15lang import update_pool
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match

def probe(local_counts, others, label):
    chains=[]
    for a,n in local_counts.items(): chains += [{"answer":a}]*n
    res=update_pool({"chains":chains}, others, None, 0.0)
    ws={a: sum(match(o,a) for o in others)/len(others) for a in local_counts}
    eps={a: 0.001*n for a,n in local_counts.items()}
    tot={a: ws[a]+eps[a] for a in local_counts}
    print(f"  {label}\n    local={local_counts} world_share={ {k:round(v,4) for k,v in ws.items()} }"
          f" total={ {k:round(v,4) for k,v in tot.items()} } -> pick {res}")
    return res

print("=== epsilon vs world_share gap ===")
print("Min nonzero world_share gap with |others|=N is 1/N. Max epsilon delta with pools<=20 is 0.001*19=0.019.\n")
for n in (14,36,40,53,60,100,200):
    print(f"  |others|={n:>4}: min nonzero gap 1/N={1/n:.4f} vs max eps delta 0.019 -> "
          f"{'CAN FLIP' if 1/n < 0.019 else 'cannot flip'}")

print("\n--- Demonstrated flip with |others|=60 (world gap 1/60=0.0167 < eps delta 0.019) ---")
others = ["100"]*31 + ["200"]*29     # 100 has strictly MORE world support
probe({"100":1, "200":20}, others, "world prefers 100 (0.5167 vs 0.4833)")

print("\n--- phaseA-realistic |others|=14: epsilon cannot flip ---")
others = ["100"]*8 + ["200"]*6
probe({"100":1,"200":20}, others, "world prefers 100 (0.5714 vs 0.4286)")

print("\n--- BUT: epsilon DOES decide every world TIE, and every zero-world case ---")
others = ["100"]*7 + ["200"]*7
probe({"100":1,"200":20}, others, "exact world tie -> local count decides")
probe({"100":20,"200":1}, others, "exact world tie, counts reversed")
print("  (with no epsilon the tie would break on the alphabetical `a` key instead)")
print("\n--- no world context at all: ws=0 for everything, epsilon is the ONLY signal ---")
probe({"100":3,"200":5}, [], "others empty -> pure local plurality")

print("\n--- world_share is NOT normalized to answers present locally; an answer with zero local")
print("    count cannot be picked at all (max over `counts` keys only). Check veto interplay: ---")
others=["300"]*13+["100"]*1   # 300 strong (0.929>=0.70), absent locally -> veto
print("  veto fires:", update_pool({"chains":[{"answer":"100"}]*5}, others, None, 0.0))
others=["300"]*9+["100"]*5    # 300 at 0.643 < 0.70 -> no veto
print("  no veto, picks local:", update_pool({"chains":[{"answer":"100"}]*5}, others, None, 0.0))
