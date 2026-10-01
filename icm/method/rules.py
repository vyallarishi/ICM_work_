"""Which version of the decision rules to run.

RECORD reproduces the headline byte for byte. V2 applies the fixes the audit
found, each one a switch so its effect can be measured on its own:

    fixed_view     vote share and plurality from the FULL pool, not the
                   8-chain view (q07_audits/fix_pool_view_share.py)
    veto_by_match  the veto's "answer not in this pool" test uses match()
                   like every other comparison, not string equality
    exact_match    answers are equal only when numerically equal; the record
                   treats answers within 0.5% as equal ("200" == "201")
    units_fallback "#### 42 dollars" parses as 42 instead of being dropped

Every function that decides takes `rules=RECORD`, so existing callers and
tests are unchanged.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    name: str
    fixed_view: bool = False
    veto_by_match: bool = False
    exact_match: bool = False
    units_fallback: bool = False


RECORD = Rules("record")
V2 = Rules("v2", fixed_view=True, veto_by_match=True, exact_match=True, units_fallback=True)
# one switch at a time, to attribute an effect
ONLY_VIEW = Rules("only_view", fixed_view=True)
ONLY_VETO = Rules("only_veto", veto_by_match=True)
ONLY_EXACT = Rules("only_exact", exact_match=True)
BY_NAME = {"record": RECORD, "v2": V2, "only_view": ONLY_VIEW, "only_veto": ONLY_VETO,
           "only_exact": ONLY_EXACT}
