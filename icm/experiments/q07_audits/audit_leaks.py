"""Item 3: can leaks() miss a leak due to number formatting?
leaks(text, answer) = any(match(norm(n), answer) for n in NUMBER_RE.findall(text))
NUMBER_RE = r"-?\d[\d,]*(?:\.\d+)?"
norm() strips commas/spaces, normalizes via Decimal.
match(a,b) = exact OR relative diff <= 0.5%"""
import sys, re
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import norm, match
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def leaks(text, answer):
    if answer is None: return False
    return any(match(norm(n), answer) for n in NUMBER_RE.findall(text))

cases = [
    # (peer text, candidate answer, SHOULD it be flagged as leaking?)
    ("The total is 1,234 dollars.",        norm("1234"),  True,  "comma thousands"),
    ("The total is 1234 dollars.",         norm("1,234"), True,  "comma in answer"),
    ("He has 72.0 apples.",                norm("72"),    True,  "trailing .0"),
    ("He has 72 apples.",                  norm("72.0"),  True,  "trailing .0 reversed"),
    ("Result: 72.00",                      norm("72"),    True,  "two decimals"),
    ("Answer 1 234 (space sep)",           norm("1234"),  True,  "SPACE thousands sep"),
    ("Answer 1.234,56 (euro style)",       norm("1234.56"),True, "EU decimal comma"),
    ("Total $1,234.50",                    norm("1234.5"),True,  "currency+comma+decimal"),
    ("It is 40 units",                     norm("40"),    True,  "plain"),
    ("<<400/10=40>>40",                    norm("40"),    True,  "calc annotation"),
    ("It is -40 below",                    norm("-40"),   True,  "negative"),
    ("It is 40 below",                     norm("-40"),   True,  "sign mismatch (abs)"),
    ("about 39.9 units",                   norm("40"),    True,  "0.25% off -> tolerance"),
    ("about 39.0 units",                   norm("40"),    False, "2.5% off -> genuinely not a leak"),
    ("The answer is 1e3",                  norm("1000"),  True,  "scientific notation"),
    ("The answer is ٤٠ (arabic-indic)",    norm("40"),    True,  "non-ASCII digits"),
    ("The answer is forty",                norm("40"),    False, "spelled out (fair miss)"),
    ("Total 12,34,567 (indian lakh)",      norm("1234567"),True, "indian grouping"),
]
print(f"{'flagged':>8} {'expect':>7}  case")
miss=[]
for text, ans, expect, label in cases:
    got = leaks(text, ans)
    ok = "OK " if got==expect else "MISS" if expect else "FP "
    if got!=expect: miss.append((label,text,ans,expect,got))
    print(f"{str(got):>8} {str(expect):>7}  {ok} {label:<26} text={text!r} ans={ans!r}")
print("\n--- discrepancies ---")
for label,text,ans,expect,got in miss:
    print(f"  {label}: expected {expect}, got {got}  (text={text!r}, ans={ans!r})")
print("\n--- what NUMBER_RE actually extracts ---")
for t in ["1,234","1 234","1.234,56","$1,234.50","1e3","٤٠","12,34,567","72.0","-40"]:
    print(f"  {t!r:>14} -> {NUMBER_RE.findall(t)}  norm-> {[norm(x) for x in NUMBER_RE.findall(t)]}")
