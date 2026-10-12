"""Run the record labelling method (icm/method with RECORD rules, every function unchanged) on a
pre-registered language set (PREREG_4):

  snapshot (CPU): copy the set's pooled chains out of the merged generation record, so a later
                  re-merge of that record cannot change them mid-run
  gen      (GPU): vote, then regenerate chains for the contested pools          -> <work>/regen.jsonl
  decide   (CPU): vote + regeneration decision                                  -> <work>/labels.json
                  (committed answer per pool, null = abstain) + coverage/precision (the only use of gold)

Language sets (results/language_selection.json): core25 = core 24 + en (the jury is these 25 only);
all67 = the 66 extension languages + en.

    python label_run.py snapshot --set core25
    python label_run.py gen      --set core25 --model-dir <staged model dir>
    python label_run.py decide   --set core25
"""
import argparse
import os
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/rvyalla/multilingual_icm/ICM_work_")
from icm.method.data import load, match  # noqa: E402
from icm.method.regenerate import build_jobs, generate, load_regen  # noqa: E402
from icm.method.rules import RECORD  # noqa: E402
from icm.method.run import decide as method_decide  # noqa: E402
from icm.method.vote import converge  # noqa: E402
from icm.experiments.q09_does_the_dataset_beat_translation.build_sft_sets import committed_answers  # noqa: E402

F = Path("/home/rvyalla/multilingual_icm/data/findings")
RECORD_ROOT = Path(os.environ.get("RECORD_ROOT", "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full66-7473"))
WORK = Path("/scratch/rvyalla/multilingual-icm/results")


def language_set(name):
    sel = json.load(open(F / "results/language_selection.json"))
    if name == "core25":
        langs = ["en"] + [l["code"] for v in sel["core"]["tiers"].values() for l in v["langs"]]
    elif name == "all67":
        langs = ["en"] + [l["code"] for v in sel["extension"]["per_tier"].values() for l in v]
    else:
        raise SystemExit(f"unknown set {name}")
    assert len(langs) == {"core25": 25, "all67": 67}[name], len(langs)
    return sorted(langs)


def work_dir(name):
    return WORK / f"labels_{name}"


def snapshot(name):
    langs = language_set(name)
    src = RECORD_ROOT / "final/pooled/generated"
    dst = work_dir(name) / "final/pooled/generated"
    dst.mkdir(parents=True, exist_ok=True)
    for l in langs:
        shutil.copy2(src / f"{l}.jsonl", dst / f"{l}.jsonl")
        n = sum(1 for _ in open(dst / f"{l}.jsonl", encoding="utf-8"))
        assert n == 7473, (l, n)
    print(f"snapshot of {len(langs)} languages x 7473 problems -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "gen", "decide"])
    ap.add_argument("--set", required=True, choices=["core25", "all67"])
    ap.add_argument("--model-dir")
    a = ap.parse_args()
    if a.cmd == "snapshot":
        return snapshot(a.set)

    langs = set(language_set(a.set))
    root = str(work_dir(a.set))
    pools = load(root, RECORD, langs)
    found = {p["lang"] for p in pools.values()}
    assert found == langs, f"pools missing for {sorted(langs - found)}"
    print(f"{len(pools)} pools over {len(found)} languages ({a.set})", flush=True)

    if a.cmd == "gen":
        state, contested, oscillators = converge(pools, rules=RECORD)
        jobs = build_jobs(pools, state, contested, RECORD)
        abstained = sum(state[i]["answer"] is None for i in contested)
        print(f"contested {len(contested)} (abstained {abstained}, oscillators {len(oscillators)}); "
              f"regen jobs {len(jobs)}", flush=True)
        generate(jobs, a.model_dir, str(work_dir(a.set) / "regen.jsonl"), RECORD)
        print(f"wrote {work_dir(a.set) / 'regen.jsonl'}")
        return

    regen = load_regen(str(work_dir(a.set) / "regen.jsonl"))
    labels = committed_answers(pools, regen)
    summary = method_decide(pools, regen, rules=RECORD)
    per_lang = defaultdict(Counter)
    for iid, ans in labels.items():
        c = per_lang[pools[iid]["lang"]]
        c["pools"] += 1
        if ans is not None:
            c["committed"] += 1
            c["correct"] += bool(match(ans, pools[iid]["gold"], RECORD))
    per_lang = {l: {"pools": c["pools"], "coverage": round(c["committed"] / c["pools"], 4),
                    "precision": round(c["correct"] / max(1, c["committed"]), 4)} for l, c in sorted(per_lang.items())}
    out = work_dir(a.set) / "labels.json"
    json.dump({"set": a.set, "languages": sorted(langs), "labels": labels}, open(out, "w"))
    rep = {"set": a.set, "n_pools": len(pools), "n_regen_pools": len(regen), "method": summary, "per_language": per_lang}
    json.dump(rep, open(F / f"results/labels_{a.set}_summary.json", "w"), indent=1)
    print(json.dumps(summary["with_regen"], indent=1))
    for l, v in per_lang.items():
        print(f"  {l:6s} coverage {v['coverage']:.4f} precision {v['precision']:.4f}")
    print(f"wrote {out} and results/labels_{a.set}_summary.json")


if __name__ == "__main__":
    main()
