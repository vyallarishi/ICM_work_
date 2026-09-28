"""AUDIT 1/D/E: weighted_decide unanimity branch, generate-constraint, veto.
Read-only: imports the real function, exercises it on synthetic pools."""
from collections import Counter
from ..q03_does_regeneration_help.regeneration_15lang import weighted_decide, WORLD_STRENGTH

fails = []
def ck(name, got, want):
    ok = got == want
    if not ok: fails.append(f"{name}: got {got!r} want {want!r}")
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  (got {got!r})")

def P(*answers):
    return {"chains": [{"answer": a} for a in answers]}

print("=== 1. unanimity_required branch semantics ===")
# Claim in docstring: "oscillator without unanimity: regen may not drive a commit"
# Case A: regen split, world answer NOT in local pool -> should abstain.
ck("A osc split, world absent from local -> NULL",
   weighted_decide(P("30"), ["99"]*10, ["99","99","98"], 1.0, True), None)
# Case B: regen split, but ONE of the regen answers equals the world answer AND
# is also already in the local pool -> rebuild drops regen, local still has it.
ck("B osc split, world answer already local -> local decides",
   weighted_decide(P("99","30"), ["99"]*10, ["99","98"], 1.0, True), "99")
# Case C: THE BUG PROBE -- regen is UNANIMOUS (len(regen_counts)==1) on an
# answer absent from the local pool, and pool is an oscillator.
ck("C osc UNANIMOUS regen on absent answer -> commits (bar does not apply)",
   weighted_decide(P("30"), ["99"]*10, ["99","99","99"], 1.0, True), "99")
# Case D: oscillator, regen EMPTY -> branch skipped entirely, veto applies
ck("D osc, no regen at all -> veto NULL",
   weighted_decide(P("30"), ["99"]*10, [], 1.0, True), None)
# Case E: oscillator, regen unanimous but on a WRONG-of-world answer
ck("E osc unanimous regen on non-world answer, world strong+absent -> NULL",
   weighted_decide(P("30"), ["99"]*10, ["77","77","77"], 1.0, True), None)

print("\n=== 1b. does the rebuild actually change anything? ===")
# The rebuild sets counts = original-pool-only. Question: can the split-regen
# case ever differ from the non-oscillator case ONLY via score tiebreak, i.e.
# is the branch doing real work or is it (mostly) a no-op?
# Probe: split regen where the regen mass would tip an EXISTING local answer.
pool = P("30","30","10")
osc_on  = weighted_decide(pool, [], ["10","10","10","20"], 1.0, True)
osc_off = weighted_decide(pool, [], ["10","10","10","20"], 1.0, False)
print(f"  split regen, no world: unanimity_required=True -> {osc_on!r}, False -> {osc_off!r}")
ck("1b rebuild suppresses regen tipping an existing answer", (osc_on, osc_off), ("30","10"))

print("\n=== 1c. INVERSION PROBE: unanimity bar vs regen COUNT ===")
# Docstring says oscillators "require regen unanimity to commit".
# But the guard is len(regen_counts) > 1, i.e. unanimity among PARSED answers.
# If only ONE regen answer parsed, it is trivially 'unanimous'.
ck("1c single parsed regen answer counts as unanimous -> commits",
   weighted_decide(P("30"), ["99"]*10, ["99"], 1.0, True), "99")
print("  -> an oscillator with 3 of 4 samples unparseable commits on n=1 'unanimity'")

print("\n=== D. generate-constraint: may a lang be assigned an answer it never generated? ===")
# The constraint should be: final answer in {answers this language produced}
# = local chains UNION regen (regen IS this language re-deriving).
# Probe 1: pure voting path.
print("  -- pure voting (update_pool) --")
from ..q01_can_voting_label_data.consensus_loop_15lang import update_pool
r = update_pool(P("30","40"), ["99"]*10, None, 0.0)
ck("D1 update_pool never returns an answer absent from local pool", r in (None,"30","40"), True)
# Probe 2: weighted_decide with regen.
r = weighted_decide(P("30"), ["99"]*10, ["99","99"], 0.5, False)
ck("D2 weighted_decide result is local-or-regen", r in ("30","99"), True)
# Probe 3: THE KEY PROBE -- can the unanimity REBUILD leave a stale key?
# counts is rebuilt from scratch, so no. But check `top_world not in counts`
# uses the REBUILT counts -- verify the veto is re-evaluated post-rebuild.
r = weighted_decide(P("30"), ["99"]*10, ["99","98"], 1.0, True)
ck("D3 veto re-checked after rebuild (regen 99 discarded) -> NULL", r, None)

print("\n=== E. veto dissolution is not special-cased ===")
# There is no `if abstained` branch; dissolution should fall out of
# `top_world not in counts` where counts includes regen.
r_before = weighted_decide(P("30"), ["99"]*10, [], 0.5, False)
r_after  = weighted_decide(P("30"), ["99"]*10, ["99"], 0.5, False)
ck("E veto NULL without regen, dissolves with regen", (r_before, r_after), (None, "99"))
# and dissolving does not force the world answer -- score still decides
r = weighted_decide(P("30","30","30"), ["99"]*7+["30"]*3, ["99"], 0.5, False)
print(f"  weak-ish world, regen adds 99: -> {r!r} (score: ws(99)=0.7 vs ws(30)=0.3)")

print("\n=== E2. alpha is inert for veto dissolution (docstring claim) ===")
for alpha in (0.01, 0.5, 1.0, 100.0):
    r = weighted_decide(P("30"), ["99"]*10, ["99"], alpha, False)
    print(f"  alpha={alpha}: {r!r}")

print()
if fails:
    print(f"{len(fails)} FAILURES / SURPRISES:")
    for f in fails: print("  " + f)
else:
    print("all probes matched expectation")
