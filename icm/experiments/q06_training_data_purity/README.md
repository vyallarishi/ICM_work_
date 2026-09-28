# training_data — how pure is the training data this produces?

The point of labelling is to train on the result: the *harvest* is every chain
that agrees with a committed answer. A contaminated chain agrees with a
committed answer that is wrong. No vote can catch it (it agrees with the
majority), so this is the one place a judge could still earn its keep.

**Answer: the judges cannot clean it** (AUC ~0.68), and the one feature that
looks like it can (world agreement, 0.926) is circular: it is nearly the label
itself. See `q07_audits/audit_world_circ.py`.

| file | was | question |
|---|---|---|
| `judge_separates_contaminated_chains.py` | harvest_evidence | can the verdict judge separate genuine from contaminated chains? 0.677 |
| `harvest_cleaning_frontier.py` | harvest_frontier | the retention / contamination curve if you filter by judge margin, per tier |
| `world_agreement_filter.py` | harvest_world | filter by how many other languages agree with the chain's answer: 0.926, but circular |
| `harvest_size_and_purity_at_scale.py` | harvest_scale | at 1,500 problems: how many chains, how pure, with and without recovered pools |

All run on cached data (CPU).
