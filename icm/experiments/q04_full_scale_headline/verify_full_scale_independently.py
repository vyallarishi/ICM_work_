"""Independent reverification of stage25_results.json from primary artifacts.
Recomputes coverage/precision/recovery from the RAW generated jsonl (own
gold lookup, own counting loops), replays the deterministic pipeline, and
checks structural invariants. Any mismatch prints MISMATCH."""
import glob, json
from collections import Counter
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match, norm
from .full_scale_37x1500 import World, converge, load
from ..q03_does_regeneration_help.regeneration_15lang import weighted_decide
from ...paths import RUN_DIR

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
saved = json.load(open(f"{RUN_DIR}/stage25_results.json"))
sr = saved["with_regen"]

# 1) independent gold + raw row accounting straight from disk
raw_gold, raw_rows, unusable = {}, 0, 0
for path in sorted(glob.glob(f"{ROOT}/final/pooled/generated/*.jsonl")):
    for line in open(path, encoding="utf-8"):
        row = json.loads(line); raw_rows += 1
        g = norm(row.get("gold_number"))
        ok_chains = sum(1 for c in row.get("candidates", [])
                        if norm(c.get("pred_number")) is not None
                        and (c.get("text") or "").strip()
                        and c.get("finish_reason") != "length")
        if g is None or ok_chains == 0:
            unusable += 1
        else:
            raw_gold[row["id"]] = g
print(f"raw rows {raw_rows}; usable {len(raw_gold)}; unusable {unusable}")
print(f"  55500 - {unusable} = {55500 - unusable} "
      f"{'== 55485 OK' if 55500 - unusable == 55485 else 'MISMATCH'}")

# 2) replay pipeline (deterministic), rebuild final state
pools = load(ROOT)
assert len(pools) == len(raw_gold), "pool count vs raw usable MISMATCH"
world = World(pools)
state, contested, osc = converge(pools, world)
regen = {json.loads(l)["iid"]: json.loads(l)["answers"]
         for l in open(f"{RUN_DIR}/stage25_regen.jsonl", encoding="utf-8")}
final = {i: dict(s) for i, s in state.items()}
for iid in contested:
    final[iid]["answer"] = weighted_decide(
        pools[iid], world.context(final, iid), regen.get(iid, []), 0.5,
        iid in osc)

# 3) recompute headline metrics with independent loops + RAW gold
committed = {i: s["answer"] for i, s in final.items() if s["answer"] is not None}
hits = sum(match(a, raw_gold[i]) for i, a in committed.items())
cov = len(committed) / len(pools)
prec = hits / len(committed)
print(f"coverage   recomputed {cov:.4f} vs saved {sr['coverage']} "
      f"{'OK' if round(cov,4)==sr['coverage'] else 'MISMATCH'}")
print(f"precision  recomputed {prec:.4f} vs saved {sr['precision']} "
      f"{'OK' if round(prec,4)==sr['precision'] else 'MISMATCH'}")

# 4) recovery block, from scratch
abst = [i for i in contested if state[i]["answer"] is None]
rec = [i for i in abst if final[i]["answer"] is not None]
rec_ok = sum(match(final[i]["answer"], raw_gold[i]) for i in rec)
print(f"abstained  recomputed {len(abst)} vs saved {sr['abstained_before']} "
      f"{'OK' if len(abst)==sr['abstained_before'] else 'MISMATCH'}")
print(f"recovered  recomputed {len(rec)} vs saved {sr['recovered']} "
      f"{'OK' if len(rec)==sr['recovered'] else 'MISMATCH'}")
print(f"rec correct recomputed {rec_ok} vs saved {sr['recovered_correct']} "
      f"{'OK' if rec_ok==sr['recovered_correct'] else 'MISMATCH'}")

# 5) structural invariants on every recovered pool
bad_support = in_pool_already = 0
for iid in rec:
    ans = final[iid]["answer"]
    if not any(match(a, ans) for a in regen.get(iid, [])):
        bad_support += 1
    if any(match(c["answer"], ans) for c in pools[iid]["chains"]):
        in_pool_already += 1
print(f"recovered w/o regen support: {bad_support} "
      f"{'OK' if bad_support==0 else 'MISMATCH'}")
print(f"recovered answers already in original pool: {in_pool_already} "
      f"(informational; veto-dissolution implies most should be absent)")

# 6) tier table
for tier in ("high", "mid", "low"):
    t_rec = [i for i in rec if gate.TIER[pools[i]["lang"]] == tier]
    t_ok = sum(match(final[i]["answer"], raw_gold[i]) for i in t_rec)
    s_t = sr["recovery_by_tier"][tier]
    ok = (len(t_rec) == s_t["recovered"] and t_ok == s_t["correct"])
    print(f"tier {tier:<5} recovered {len(t_rec)}/{t_ok} correct vs saved "
          f"{s_t['recovered']}/{s_t['correct']} {'OK' if ok else 'MISMATCH'}")

# 7) gold-blindness spot check: no regen answer file contains gold field
sample_line = open(f"{RUN_DIR}/stage25_regen.jsonl").readline()
print(f"regen cache fields: {sorted(json.loads(sample_line).keys())} "
      f"(must not contain gold)")
print("DONE")
