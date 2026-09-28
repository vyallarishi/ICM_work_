import sys
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from collections import defaultdict
from ..q01_can_voting_label_data import consensus_loop_15lang as s9
from ..q04_full_scale_headline import full_scale_37x1500 as s25

def mkpools(spec):
    # spec: iid -> lang
    return {iid:{"lang":l,"chains":[],"gold":"1","question":"q","share":1.0,"plurality":"1"}
            for iid,l in spec.items()}

def cmp(name, pools, state, iid):
    try: a=s9.world_context(pools,state,iid)
    except Exception as e: a=f"EXC:{type(e).__name__}"
    w=s25.World(pools)
    try: b=w.context(state,iid)
    except Exception as e: b=f"EXC:{type(e).__name__}"
    same = (sorted(a)==sorted(b)) if isinstance(a,list) and isinstance(b,list) else (a==b)
    print(f"{'OK ' if same else 'DIFF'} {name}\n     stage9={a}\n     stage25={b}")
    return same

# 1) baseline
pools=mkpools({"t_en_1":"en","t_de_1":"de","t_hi_1":"hi","t_en_2":"en"})
state={i:{"answer":{"t_en_1":"10","t_de_1":"20","t_hi_1":"30","t_en_2":"40"}[i],"kind":"contested"} for i in pools}
cmp("basic", pools, state, "t_en_1")

# 2) same language appearing twice for the same problem
pools=mkpools({"t_en_1":"en","t_enX_1":"en","t_de_1":"de"})
state={"t_en_1":{"answer":"10"},"t_enX_1":{"answer":"99"},"t_de_1":{"answer":"20"}}
cmp("duplicate language same problem", pools, state, "t_en_1")

# 3) None answers
pools=mkpools({"t_en_1":"en","t_de_1":"de","t_hi_1":"hi"})
state={"t_en_1":{"answer":"10"},"t_de_1":{"answer":None},"t_hi_1":{"answer":"30"}}
cmp("None answers skipped", pools, state, "t_en_1")

# 4) state has an iid NOT in pools
pools=mkpools({"t_en_1":"en","t_de_1":"de"})
state={"t_en_1":{"answer":"10"},"t_de_1":{"answer":"20"},"t_fr_1":{"answer":"77"}}
cmp("state key missing from pools", pools, state, "t_en_1")

# 5) pools has an iid NOT in state  <-- the asymmetry
pools=mkpools({"t_en_1":"en","t_de_1":"de","t_fr_1":"fr"})
state={"t_en_1":{"answer":"10"},"t_de_1":{"answer":"20"}}   # t_fr_1 absent
cmp("pools key missing from state", pools, state, "t_en_1")

# 6) problem id parsing: multi-underscore language codes (zh-tw ok, but what about ids)
pools=mkpools({"train_zh-tw_5":"zh-tw","train_en_5":"en"})
state={i:{"answer":"1"} for i in pools}
cmp("hyphenated lang code", pools, state, "train_en_5")

# 7) ordering of returned list
pools=mkpools({"t_en_1":"en","t_de_1":"de","t_hi_1":"hi","t_ru_1":"ru"})
state={"t_en_1":{"answer":"1"},"t_de_1":{"answer":"2"},"t_hi_1":{"answer":"3"},"t_ru_1":{"answer":"4"}}
a=s9.world_context(pools,state,"t_en_1"); b=s25.World(pools).context(state,"t_en_1")
print(f"{'OK ' if a==b else 'ORDER-DIFF'} order: stage9={a} stage25={b}")
