# examples

`walkthrough.py` runs the whole method on 5 languages × 3 problems and prints
every decision with its reason. It uses the real functions from `icm/method`;
nothing about the outcome is hard-coded.

```bash
python examples/walkthrough.py
```

What to look for in the output:

- **p1**: every language agrees → all five pools commit on their own.
- **p2**: `sw` is split → the world decides. `am` is confident and wrong →
  vetoed, then abstains (the world's answer is not in its pool). After
  regeneration it produces 42 and is recovered.
- **p3**: `hi` and `am` have the wrong majority but the right answer somewhere
  in the pool → the world corrects both.

The only step that needs a model is the regeneration sampling; the script
pretends the model answered.
