# search — would a search over labelings find better labels?

**Answer: no.** The ICM paper finds labels by simulated annealing over an
energy function. We tested whether any such energy exists here. Both energies
score the *gold* labeling worse than the labeling voting already reaches, and
a downhill single flip lands on gold 0.9% of the time. The remaining errors
are cases where the languages agree with each other, so no search over
agreement can fix them.

| file | was | energy tested | result |
|---|---|---|---|
| `static_overlap_energy.py` | stage29_energy_probe | answer agreement + overlap of intermediate values between chains | dead: E(gold) > E(converged) |
| `rederivation_energy.py` | stage30_rederiv_energy | the truncated-peer re-derivation score from `q02_can_the_model_judge_itself/truncated_peer_rederivation_at_scale.py` | dead the same way |
| `threshold_schedules.py` | threshold_anneal | not a search, but the closest thing: schedule the two 0.70 thresholds across voting rounds (loosen, tighten, split) | none beats flat 0.70 on both coverage and precision |

All three run on cached data (CPU).
