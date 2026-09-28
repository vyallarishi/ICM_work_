import json
from collections import defaultdict
from .verdict_judge_gate import load_pools, match
from ...paths import RUN_DIR
pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
rows = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl", encoding="utf-8")]
# stage8_chains.jsonl dropped text; re-join margins to chains by (iid, is_gold) order
# Instead: recompute the sample the same way stage8 did, then attach margins in order.
import random
from .verdict_judge_gate import gate_sample, build_demo_bank, SEED
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
print(f"recomputed sample {len(sample)} vs saved {len(rows)}")
if len(sample) == len(rows):
    for s, r in zip(sample, rows):
        s["J3"] = r.get("J3"); s["contested"] = r.get("contested")
    # ENGLISH correct-answer chains only, so I can read them
    eng = [s for s in sample if s["lang"] == "en" and s["is_gold"] and s.get("J3") is not None]
    eng.sort(key=lambda s: s["J3"])
    print(f"english gold chains with margins: {len(eng)}")
    out = {"low": eng[:6], "high": eng[-6:]}
    for band in ("low", "high"):
        for s in out[band]:
            print(f"\n===== {band.upper()} margin {s['J3']:+.3f}  [{s['instance_id']}] gold={pools[s['instance_id']]['gold']}")
            print(s["text"][:700])
