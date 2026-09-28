# regeneration — does letting a stuck language re-solve help?

**Answer: yes.** After voting, some pools abstain: the other languages
strongly agree on an answer this language never produced. Showing the language
three peer solutions and letting it solve the problem 4 more times recovers
most of them. At full scale: 826 of 1,240 abstentions recovered, 754 correct
(91.3%), with zero correct answers withdrawn.

| file | was | what it does |
|---|---|---|
| `regeneration_15lang.py` | stage22_regen | **the mechanism** (`select_regen_peers`, `build_regen_prompt`, `weighted_decide`). 15 languages × 150: 51 of 79 recovered at 98%, coverage 96.49 → 98.76 |
| `regeneration_37lang.py` | stage23_regen37 | the same at 37 languages × 150 |
| `regeneration_seed_sweep.py` | stage24_seeds | 5 sampling seeds. **Note: covers the 150-problem root only, not the headline** |
| `regeneration_vs_blind_overwrite.py` | regen_control | control: is recovery just "trust the world"? Compares with overwriting by the world answer |
| `regeneration_failure_autopsy.py` | regen_failures | the 5 failures at 37 × 150, one by one |
| `regeneration_who_does_it_help.py` | regen_impact | what fraction of pools regeneration reaches, changes, and helps |
| `regeneration_presence_threshold.py` | regen_threshold | require a new answer T times before it counts (T = 1..4) |
| `recovery_precision_gates.py` | recovery_gate | recovery precision by tier (low 94%, mid 89%, **high 69%**) and label-blind gates to protect the high tier |
| `truncated_peer_recovery_estimate.py` | rederiv_recovery_est | would truncated peers (see judges/) recover more purely? An upper-bound sketch |

Design choices in the mechanism, and why (from `regeneration_15lang.py`):

- **Late seeding**: regenerate from the converged state, when the world is
  95.5% right, not from the start (88.7%).
- **Minority peer**: 2 peers from the majority answer + 1 from the largest
  minority, so the prompt arbitrates instead of amplifying.
- **Half weight**: new chains vote at 0.5, so 4 of them cannot outvote a pool.
- **Oscillator bar**: a pool that changed answer ≥ 2 times while converging
  must regenerate unanimously (18 of 20 such pools had ended wrong).
- **Single round**, and regenerated chains never become demos or training data.

The open problem is the high tier: a high-resource language that abstains is
usually right and the world is wrong. See `recovery_precision_gates.py`.
