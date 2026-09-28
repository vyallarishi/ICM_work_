import sys
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from decimal import Decimal, getcontext, InvalidOperation
from ..q02_can_the_model_judge_itself import verdict_judge_gate as g
from ..q08_earlier_8b_study import rung3_reasoning_structure as r
print("decimal context prec =", getcontext().prec, " Emax =", getcontext().Emax)
print("\nThreshold hunt: norm('1' + '0'*k)")
for k in range(24,32):
    s = "1"+"0"*k
    a=g.norm(s)
    try: b=r.normalize_number(s)
    except Exception as e: b="EXC:"+type(e).__name__
    print(f"  10^{k:<3} digits={k+1:<3} stage8={str(a)[:40]:<42} rung3={str(b)[:40]}")
print("\nSame for a non-round large integer, 28..30 digits:")
for k in range(26,31):
    s = "1234567890"*3
    s = s[:k]
    a=g.norm(s)
    try: b=r.normalize_number(s)
    except Exception as e: b="EXC:"+type(e).__name__
    print(f"  digits={k:<3} value={s:<32} stage8={str(a)[:34]:<36} rung3={str(b)[:34]}")
print("\nWhy: after normalize(), 1e28 has coefficient 1 exp 28; quantize(Decimal(1)) needs 29 sig digits > prec 28")
d=Decimal("1"+"0"*28).normalize()
print("  normalize() ->", d, " as_tuple", d.as_tuple())
try:
    print("  quantize ->", d.quantize(Decimal(1)))
except InvalidOperation as e:
    print("  quantize raises InvalidOperation:", e)
