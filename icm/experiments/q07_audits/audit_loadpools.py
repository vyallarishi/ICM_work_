import sys, glob, json
from pathlib import Path
from collections import Counter, defaultdict
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself import verdict_judge_gate as g

ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
stats=defaultdict(Counter)
norm_none_examples=defaultdict(list)
for path in sorted(glob.glob(f"{ROOT}/final/pooled/generated/*.jsonl")):
    lang=Path(path).stem
    for line in open(path,encoding="utf-8"):
        row=json.loads(line)
        s=stats[lang]
        s["rows"]+=1
        gold=g.norm(row.get("gold_number"))
        if gold is None:
            s["rows_dropped_no_gold"]+=1
        cands=row.get("candidates",[])
        s["cands_total"]+=len(cands)
        kept=0
        for c in cands:
            raw=c.get("pred_number")
            a=g.norm(raw)
            txt=c.get("text") or ""
            if raw is None: s["cand_null_pred"]+=1
            elif a is None:
                s["cand_norm_failed"]+=1
                if len(norm_none_examples[lang])<4: norm_none_examples[lang].append(raw)
            if a is None or not txt.strip():
                s["cand_dropped_answer_or_text"]+=1; continue
            if c.get("finish_reason")=="length":
                s["cand_dropped_length"]+=1; continue
            kept+=1
        s["cands_kept"]+=kept
        if kept==0: s["rows_dropped_no_chains"]+=1
        if kept>0 and gold is not None: s["pools_kept"]+=1

hdr=f"{'lang':<6}{'rows':>6}{'pools':>7}{'%pool':>7}{'cands':>8}{'kept':>8}{'%kept':>7}{'drop_len':>9}{'%len':>7}{'nullpred':>9}{'normfail':>9}{'no_gold':>8}"
print(hdr)
for lang in sorted(stats):
    s=stats[lang]
    print(f"{lang:<6}{s['rows']:>6}{s['pools_kept']:>7}{100*s['pools_kept']/s['rows']:>6.1f}%"
          f"{s['cands_total']:>8}{s['cands_kept']:>8}{100*s['cands_kept']/max(s['cands_total'],1):>6.1f}%"
          f"{s['cand_dropped_length']:>9}{100*s['cand_dropped_length']/max(s['cands_total'],1):>6.1f}%"
          f"{s['cand_null_pred']:>9}{s['cand_norm_failed']:>9}{s['rows_dropped_no_gold']:>8}")
print("\nnorm() failures (raw pred_number values that normed to None):")
for lang,ex in sorted(norm_none_examples.items()):
    if ex: print(f"  {lang}: {ex}")

pools=g.load_pools(ROOT)
print(f"\nload_pools -> {len(pools)} pools; langs {sorted({p['lang'] for p in pools.values()})}")
per=Counter(p['lang'] for p in pools.values())
print("pools per lang:", dict(sorted(per.items())))
sizes=Counter(len(p['chains']) for p in pools.values())
print("chains-per-pool distribution:", dict(sorted(sizes.items())))
# per-lang mean chains
mc=defaultdict(list)
for p in pools.values(): mc[p['lang']].append(len(p['chains']))
print("\nmean chains/pool by lang (pool size drives share/landslide threshold):")
for l in sorted(mc):
    import statistics
    print(f"  {l}: n={len(mc[l])} mean={statistics.mean(mc[l]):.2f} min={min(mc[l])} max={max(mc[l])}")
