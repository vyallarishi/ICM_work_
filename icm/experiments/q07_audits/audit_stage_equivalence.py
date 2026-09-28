"""AUDIT 4/G: converge() vs converge_w0() equivalence, and what stage24 covers."""
import json
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q01_can_voting_label_data.consensus_loop_15lang import metrics, world_context
from ..q03_does_regeneration_help.regeneration_15lang import converge_w0, weighted_decide
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from ...paths import RUN_DIR

OLD="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"
print("=== 4. converge() [stage25] vs converge_w0() [stage22/23] on 150-prob root ===")
pools=load(OLD)
w=World(pools)
s25,c25,o25=converge(pools,w)
pools2=s17.apply_diverse_views(s17.load_pools(OLD))
s22,c22,o22=converge_w0(pools2)
print(f"  contested: stage25={len(c25)} stage22={len(c22)}  same={len(c25)==len(c22)}")
print(f"  oscillators: stage25={len(o25)} stage22={len(o22)}  same={o25==o22}")
same=all(s25[i]["answer"]==s22[i]["answer"] for i in s25)
print(f"  every pool same converged answer: {same}")
print(f"  metrics stage25: {metrics(pools,s25,c25)}")
print(f"  metrics stage22: {metrics(pools2,s22,c22)}")
print("  -> converge() and converge_w0() ARE the same 4-lap procedure "
      "(both synchronous, both 4 laps, same oscillator rule).")

print("\n=== 4b. BUT the post-regen decide step differs across stages? ===")
print("  stage22:311-315  others = world_context(pools, new_state, iid)   SEQUENTIAL")
print("  stage23:146-150  others = world_context(pools, new_state, iid)   SEQUENTIAL")
print("  stage24:70-74    others = world_context(pools, new_state, iid)   SEQUENTIAL")
print("  stage25:174-178  others = world.context(new_state, iid)          SEQUENTIAL")
print("  -> all four are consistently sequential. Comparable to each other.")
print("  BUT converge()/converge_w0() laps are SYNCHRONOUS (read frozen `state`).")
print("  So the regen lap uses different update semantics than every converge lap.")

print("\n=== G. what root does stage24 actually validate? ===")
d=json.load(open(f"{RUN_DIR}/stage24_results.json"))
from ..q03_does_regeneration_help import regeneration_seed_sweep as s24
print(f"  stage24 ROOTS = {s24.ROOTS}")
print(f"  -> 37lang arm = qwen3-14b-full37  (150 problems), NOT full37-1500.")
print(f"  stage25 headline root = qwen3-14b-full37-1500 (1500 problems).")
print(f"  stage24 37lang abstained_before = {d['37lang']['per_seed'][0]['abstained_before']}"
      f"  vs stage25 abstained_before = 1240")
print("  -> seed robustness was NEVER run on the headline 1500-problem pipeline.")

print("\n=== G2. stage24 pre-registration vs its own result ===")
print("  docstring: 'seed-robust if recovery precision stays >= 95% at EVERY seed'")
for r in d["37lang"]["per_seed"]:
    ok = r["recovery_precision"]>=0.95
    print(f"    seed {r['seed']}: rec_prec {r['recovery_precision']} "
          f"{'PASS' if ok else 'FAIL pre-registered bar'}")
print("  -> 3 of 5 seeds MISS the pre-registered >=95% bar (92.5, 93.83, 94.94).")
