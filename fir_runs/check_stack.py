"""Fir vs Rorqual software-stack check: every Fir evaluation also scores the BASE model, which Rorqual scored in
wave 1 (fir_runs/reference/core24_<student>_<set>_gold.json). Identical stacks give identical base accuracy.
usage: python check_stack.py   (after at least one Fir evaluation has finished)"""
import glob, json, os
H = os.path.dirname(os.path.abspath(__file__))
R = os.path.expanduser("~/multilingual_icm/data/findings/results")
for ref in sorted(glob.glob(f"{H}/reference/core24_*_gold.json")):
    stem = os.path.basename(ref)[:-len("_gold.json")]
    for fir in sorted(glob.glob(f"{R}/{stem}_*.json")):
        if fir.endswith("_gold.json"):
            continue
        a = json.load(open(ref))["arms"]["base"]["accuracy"]; b = json.load(open(fir))["arms"]["base"]["accuracy"]
        keys = sorted(set(a) & set(b)); diff = max(abs(a[k] - b[k]) for k in keys)
        print(f"{os.path.basename(fir):55s} base ALL Rorqual {a['ALL']:.4f}  Fir {b['ALL']:.4f}  max per-language diff {diff:.4f}"
              + ("  OK" if diff <= 0.005 else "  CHECK: stacks differ"))
