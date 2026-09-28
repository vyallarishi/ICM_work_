import sys, random
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from decimal import Decimal, InvalidOperation
from ..q02_can_the_model_judge_itself import verdict_judge_gate as g
from ..q08_earlier_8b_study import rung3_reasoning_structure as r

cases = ["2190.0","1,200"," 12 ","0","-0","0.0","-0.0","1e5","1E+5","1e-5","2e30",
         "1e100000","1e-100000","nan","NaN","inf","-Infinity","snan","abc","",
         None, 12, 12.5, True, "0.005","1.0000000001","3.14159",
         "1"+"0"*50, "1e999999999", "0."+"0"*40+"1", "  1,2 3 4 ", "+5", "--5",
         "1_000", "٣", "１２３", "1.5e3", "0e0"]
print(f"{'input':<24}{'stage8.norm':<28}{'rung3.normalize_number':<28}{'agree'}")
div=[]
for c in cases:
    try: a=g.norm(c)
    except Exception as e: a=f"EXC:{type(e).__name__}"
    try: b=r.normalize_number(c)
    except Exception as e: b=f"EXC:{type(e).__name__}"
    ok = (a==b)
    if not ok: div.append((c,a,b))
    print(f"{str(c):<24}{str(a):<28}{str(b):<28}{ok}")
print("\nDIVERGENCES:")
for c,a,b in div: print("  ",repr(c),"stage8=",repr(a)," rung3=",repr(b))

print("\n--- match() vs answers_match() ---")
pairs=[("2190","2190"),("10","9.9996"),("9.9996","10"),("0","0"),("0","0.0000001"),
       ("-10","-9.99"),("100","100.4"),("100","100.6"),("1E+5","100000"),
       (None,"1"),("1",None),("-5","5"),("0","-0")]
for x,y in pairs:
    try: m=g.match(x,y)
    except Exception as e: m=f"EXC:{type(e).__name__}"
    try: n=r.answers_match(x,y)
    except Exception as e: n=f"EXC:{type(e).__name__}"
    print(f"  match({x!r},{y!r}) stage8={m} rung3={n} agree={m==n}")

print("\n--- match() on raw (non-normed) strings: does stage8.match crash? ---")
for bad in ["nan","inf","abc","1,200",""]:
    try:
        print("  match(",repr(bad),",'1') ->", g.match(bad,"1"))
    except Exception as e:
        print("  match(",repr(bad),",'1') -> EXC", type(e).__name__, e)
    try:
        print("  answers_match(",repr(bad),",'1') ->", r.answers_match(bad,"1"))
    except Exception as e:
        print("  answers_match(",repr(bad),",'1') -> EXC", type(e).__name__, e)
