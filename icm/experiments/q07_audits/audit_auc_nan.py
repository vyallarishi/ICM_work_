import sys, signal
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import auc
print("all tied:",auc([1,1,1],[1,1]))
print("empty:",auc([],[1]),auc([1],[]))
def handler(s,f): raise TimeoutError("auc did not return")
signal.signal(signal.SIGALRM, handler); signal.alarm(5)
try:
    print("nan:", auc([float('nan'),1.0],[0.0]))
except TimeoutError as e:
    print("NaN case: INFINITE LOOP / hang ->", e)
signal.alarm(0)
# why: NaN == NaN is False, so inner while never advances past a NaN run? investigate
import itertools
n=float('nan')
ranked=sorted([(0.0,0),(n,1),(1.0,1)], key=lambda t:t[0])
print("sorted with nan:", ranked)
i=0; stop=i
print("ranked[stop][0]==ranked[index][0] for nan:", ranked[1][0]==ranked[1][0])
