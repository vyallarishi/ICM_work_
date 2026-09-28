# Multilingual ICM

Getting correct answers to maths problems **without any human labels**.

The model (Qwen3-14B-Base) solves each GSM8K problem about 20 times in each of
37 languages. Where the solutions in one language agree, we keep that answer.
Where they are split, the other 36 languages vote. Where a language cannot
produce the answer the others agree on, it is shown three of their solutions
and solves the problem again.

**Result on 37 languages × 1,500 problems (55,485 pools):
99.23% of pools get an answer, and 94.45% of those answers are correct.**
Gold answers are used only to measure this, never to decide anything.

## Start here

1. **See it work** (1 second, no data needed):
   ```bash
   python examples/walkthrough.py
   ```
2. **Read the method**: [icm/method/](icm/method/) — four short files, in the
   order `data.py` → `vote.py` → `regenerate.py` → `metrics.py`.
3. **Understand the words**: [GLOSSARY.md](GLOSSARY.md).
4. **Go deeper**: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) explains the
   decision rule step by step, plus known issues and what to improve next.

## What is where

| folder | what it is |
|---|---|
| [generation/](generation/) | Step 1: sampling the ~1M solution chains (GPU, vLLM). Has its own README. |
| [icm/method/](icm/method/) | Step 2: the method itself. 
| [icm/experiments/](icm/experiments/) | Every experiment we ran, grouped by the question it asked. Each folder has a README with the answer. |
| [examples/](examples/) | The toy walkthrough. |
| [recorded_results/RESULTS.md](recorded_results/RESULTS.md) | What every result file shows. The files themselves are in `data/ICM_work_/recorded_results/`, outside git. |
| [tests/](tests/) | Proves `icm/method` gives exactly the same output as the experiment code. |
| [tools/](tools/) | Checks that the experiment files are unchanged from the originals. |
| `data/ICM_work_/runs/` | Working directory for re-runs (not in git). |

## Running things

The method needs only Python 3.10+ (no extra packages). Generation needs vLLM;
see [generation/README.md](generation/README.md).

```bash
python -m unittest discover -s tests            # method == experiments (5 s)
python -m icm.method.run decide                 # reproduce the headline (needs the pools)
sbatch icm/experiments/q04_full_scale_headline/reproduce_headline.sbatch   # the same, as a cluster job
```

Every `.sbatch` job runs from `data/ICM_work_/runs/` (the scripts open their caches relative
to the current directory, as they did in `~/phase0`); the method's own paths
are anchored, so `python -m icm.method.run` works from anywhere.

Reproducing any number needs the generated pools on `/scratch` and the caches
in `data/ICM_work_/runs/`, which are not in this repository. Everything else runs anywhere.

