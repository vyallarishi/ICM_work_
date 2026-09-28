"""Is the P3 gap confounded by PEER COUNT?
settled arms: peers_for_answer  -> requires the other language's COMMITTED
              answer to match  => consensus answers get MANY peers
wrong arms:   peers_supporting  -> only requires a CHAIN to match
              => wrong answers get FEWER / lower-quality peers
Fewer peers in context => weaker conditioning => lower S, independent of
whether the derivation is 'sound'. Measure the peer-count distributions."""
import sys, random, json
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from collections import Counter, defaultdict

# Use the stage27 (15-lang phaseA) setting: cheap enough to reconstruct.
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from ..q01_can_voting_label_data.consensus_loop_15lang import initialize, update_pool
from ..q04_full_scale_headline.full_scale_37x1500 import World
from ..q02_can_the_model_judge_itself.generative_mutual_predictability import candidates_for, peers_for_answer
from ..q02_can_the_model_judge_itself.truncated_peer_rederivation import peers_supporting, wrong_candidate, truncate_peer, SEED, ITERATIONS

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
world = World(pools)
state = initialize(pools)
contested = [i for i,s in state.items() if s["kind"]=="contested"]
for _ in range(ITERATIONS):
    ns={i:dict(s) for i,s in state.items()}
    for iid in contested:
        ns[iid]["answer"]=update_pool(pools[iid], world.context(state,iid), None, 0.0)
    state=ns
print(f"pools {len(pools)} contested {len(contested)}")

rng=random.Random(SEED)
set_counts=[]; wrong_counts=[]
for iid in contested:
    for answer in candidates_for(pools, state, world, iid):
        p=peers_for_answer(pools, state, iid, answer, rng)
        if p: set_counts.append(len(p))
    alt=wrong_candidate(pools, state, world, iid)
    if alt is not None:
        ap=peers_supporting(pools, iid, alt, rng)
        if ap: wrong_counts.append(len(ap))

print("\n=== PEER COUNT per prompt ===")
print("  settled arms:", Counter(set_counts).most_common(), 
      f" mean={sum(set_counts)/len(set_counts):.3f}")
print("  wrong arms:  ", Counter(wrong_counts).most_common(),
      f" mean={sum(wrong_counts)/len(wrong_counts):.3f}")
