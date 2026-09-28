import sys, time
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import auc
t=time.time(); print(auc([1,2],[0,3]), time.time()-t)
t=time.time(); print(auc([0,0,0],[0,0]), time.time()-t)
t=time.time(); print(auc([0.0]*50,[0.0]*50), time.time()-t)
t=time.time(); print(auc([0.0]*200,[0.0]*200), time.time()-t)
