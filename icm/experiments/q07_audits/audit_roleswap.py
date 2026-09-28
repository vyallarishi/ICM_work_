"""Item 1: role-swap sign convention. Simulate the exact code path."""
# stage8/18/19/21 pattern:
#   for role_swap in (False, True):
#       positive, negative = (" No"," Yes") if role_swap else (" Yes"," No")
#       pos_id, neg_id = enc(positive), enc(negative)
#       margin = logp[pos_id] - logp[neg_id]
#   final = (norm + swap)/2
#
# In the swap pass positive==" No", so pos_id is the NO token.
# margin_swap = logP(" No") - logP(" Yes")   <-- measured on the SWAPPED PROMPT
# The prompt also changes: the instruction says mark with " No" if correct.
# So on the swapped prompt, "correct" maps to the " No" token.
# Therefore margin_swap is ALSO oriented "higher = judged correct". Correct.

import random
rng = random.Random(0)

def simulate(chain_is_good, yes_bias, swap_faithful=True, invert_bug=False):
    """Model emits higher logprob for whichever token the PROMPT designates
    as 'correct', plus a fixed position/identity bias toward the Yes token."""
    signal = 1.0 if chain_is_good else -1.0
    # normal prompt: 'correct' marker = Yes
    lp_yes_n = signal + yes_bias
    lp_no_n  = -signal
    m_norm = lp_yes_n - lp_no_n            # pos=Yes, neg=No
    # swapped prompt: 'correct' marker = No
    lp_no_s  = signal
    lp_yes_s = -signal + yes_bias
    if invert_bug:
        # the BUG would be: keep pos_id=Yes on the swapped prompt
        m_swap = lp_yes_s - lp_no_s
    else:
        m_swap = lp_no_s - lp_yes_s        # pos=No, neg=Yes  (as coded)
    return (m_norm + m_swap) / 2

print("=== correct-as-coded convention (pos_id follows `positive`) ===")
for bias in (0.0, 3.0, -3.0):
    g = simulate(True, bias); b = simulate(False, bias)
    print(f"  yes_bias={bias:+.1f}  good={g:+.3f} bad={b:+.3f}  separation={g-b:+.3f}")

print("=== hypothetical inverted swap (pos_id pinned to Yes) ===")
for bias in (0.0, 3.0, -3.0):
    g = simulate(True, bias, invert_bug=True); b = simulate(False, bias, invert_bug=True)
    print(f"  yes_bias={bias:+.1f}  good={g:+.3f} bad={b:+.3f}  separation={g-b:+.3f}")
