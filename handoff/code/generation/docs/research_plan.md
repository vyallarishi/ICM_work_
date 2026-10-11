# Multilingual ICM Experimental Protocol

## Primary Claim

The primary experiment studies target-label-blind concept discovery from naturally sampled base
model reasoning chains. Gold answers are used for post-hoc analysis and evaluation, not for choosing
the candidates supplied to ICM.

Because native solved demonstrations are used during generation, the few-shot experiment is not
globally unsupervised. The precise claim is **target-label-blind concept discovery with in-context
supervision**. The zero-shot ablation removes that in-context supervision.

## Generation

Use Qwen3-8B-Base to sample 20 candidates per train item:

- `target_low_temp`: native six-shot CoT, temperature 0.2, `n=8`.
- `target_mid_temp`: native six-shot CoT, temperature 0.7, `n=8`.
- `target_high_temp`: native six-shot CoT, temperature 1.0, `n=4`.

The native demonstrations use test-split `original_id` values `(0, 5, 7, 10, 13, 20)`. These
demonstration items must not later be reported as held-out evaluation examples.

Keep every raw regime output. Parsing a generated final answer and recording its correctness is
permitted for audit and post-hoc analysis, but those fields must be absent from ICM input files.

## Candidate Views

Run the same downstream analysis using these named views:

| View | Target gold used to construct it? | Purpose |
|---|---:|---|
| `all_usable` | No | Primary, maximum available evidence |
| `random_4` | No | Compute-matched natural-subset baseline |
| `diverse_4` | No | Conflict-enriched label-blind baseline |
| `oracle_balanced_4` | Yes | Supervised upper bound/control |
| wrong-chain ablation | Synthetic instruction | Artificial-conflict ablation |

Quality filtering may use answer parseability, length, repetition, and exact/near-exact duplicate
text. It must not use correctness. `diverse_4` may use the model's predicted answer value, but not a
comparison with the reference answer.

## Retry Policy

The primary retry policy is label-blind:

- Retry low-temperature native few-shot generation when fewer than four usable candidates exist.
- Retry high-temperature native few-shot generation when fewer than two predicted-answer clusters
  exist.

Do not trigger retries using `no_correct`, `no_incorrect`, or any other gold-derived property. Keep
retry provenance so analyses can compare main-only candidates with expanded pools.

## Reporting Checklist

Report before ICM training:

- Raw, quality-passing, usable, and deduplicated candidate counts by language.
- Correct and incorrect usable counts and post-hoc candidate accuracy by language.
- Fraction of items with fewer than four usable chains.
- Fraction with no correct or no incorrect chain, for analysis only.
- Number of distinct predicted answers per item.
- Main versus retry candidate provenance.
- Manual chain-quality audits stratified across scripts and language families.

For every result, state which candidate view was used. Compare `all_usable`, `random_4`, and
`diverse_4` before interpreting `oracle_balanced_4`.

## Scientific Risks

Final-answer correctness does not establish reasoning correctness. Some numerically correct chains
will contain invalid reasoning, while incorrect chains may contain useful intermediate concepts.
Manually annotate a stratified subset or use an independently validated process verifier for
reasoning-level conclusions.

Few-shot and temperature mixtures modify the base model's output distribution. Preserve regime
metadata and include the zero-shot ablation. Do not describe a conflict-enriched diverse view as an
unmodified natural sample.

Using the same model for data generation and fine-tuning can amplify model-specific biases. Retain a
strict held-out evaluation set and compare against independent-model or cross-model evaluation where
possible.
