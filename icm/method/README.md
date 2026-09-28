# icm/method — the method itself

Four short files. Read them in this order:

| file | what it does |
|---|---|
| `data.py` | reads the generated chains into pools; `norm` and `match` compare answers |
| `vote.py` | `initialize`: commit / contest / veto. `converge`: contested pools re-vote for 4 rounds |
| `regenerate.py` | a stuck language sees three peer solutions and tries again; `weighted_decide` re-decides |
| `metrics.py` | coverage and precision. The only file that reads gold |
| `run.py` | puts them together: `python -m icm.method.run decide` |

Every function is a copy of the one in the experiment that produced the
headline (`icm/experiments/q04_full_scale_headline/full_scale_37x1500.py` and the files it
imports). The only deliberate difference: `update_pool` has no judge-bonus
parameters, because no judge was ever used in the final method.

`tests/test_method_matches_experiments.py` runs both code paths on a synthetic
dataset and requires identical output at every step. Run it after any change:

```bash
python -m unittest discover -s tests
```

`icm/experiments/q04_full_scale_headline/reproduce_headline.sbatch` does the same on the
real data.

Constants worth knowing (`data.py`, `regenerate.py`):

| name | value | meaning |
|---|---|---|
| `LANDSLIDE` | 0.70 | a pool commits alone when this share of its chains agree |
| `WORLD_STRENGTH` | 0.70 | the other languages "strongly agree" at this share |
| `ITERATIONS` | 4 | voting rounds |
| `M_SAMPLES` | 4 | new chains per stuck pool |
| `ALPHA` | 0.5 | weight of a new chain |
