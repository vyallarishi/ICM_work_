# voting — can languages voting on each other label the data?

**Answer: yes.** Pure iterated consensus (no model judge) reaches 98.02%
coverage at 95.22% precision on 37 languages × 150 problems. Adding a judge
to the loop gives at most +2.6 points on contested pools at one weight and
collapses at higher weights, so the shipped method has no judge.

Data: `qwen3-14b-phaseA` (15 languages × 150) and `qwen3-14b-full37` (37 × 150).

## The loop and its views

| file | was | what it does |
|---|---|---|
| `consensus_loop_15lang.py` | stage9_loop | **The decision rule** (`initialize`, `update_pool`, `metrics`) plus the first experiment: three arms, judge weight 0 / 0.5 / 7. Result: w0 86.6% contested accuracy, light judge 89.2%, heavy judge collapses. |
| `consensus_loop_diverse4_views.py` | stage9d_diverse | the same loop on 4-chain views |
| `consensus_loop_37lang.py` | stage17_diverse8_37 | the same at 37 languages with 8-chain views. **Defines `diverse8_view` and extends the language table to 37** (patches stage 8's table on import). |
| `diverse8_vs_diverse4_cpu.py` | stage12_diverse8_cpu | why 8-chain views: gold coverage and judge window vs 4-chain and full views |
| `judge_actionable_window_37lang.py` | window37 | how many pools a judge could even flip at 37 languages |

## Trying harder to make the judge matter (all superseded)

| file | was | tried | result |
|---|---|---|---|
| `judge_weight_annealed.py` | stage10_sched | judge weight rising 0.5 → 4 across 6 laps | no plateau break |
| `judge_weight_annealed_diverse4.py` | stage10d_diverse | the same on 4-chain views | judge went silent |
| `why_judge_went_silent_on_diverse4.py` | stage11_judge_autopsy | one chain per answer starves the judge; 8-chain views fix it | led to diverse-8 |
| `judge_weight_annealed_diverse8.py` | stage13_diverse8 | annealed weight on 8-chain views | |
| `judge_first_init_synthesis.py` | stage15_final | contested pools initialised from the judge's proposal, then annealed trust | |
| `judge_first_init_cpu_sim.py`, `judge_first_init_decomposition.py` | stage15_cpu_sim, stage15_decomp | CPU simulations: where does the gain come from? | ordering vs annealing |
| `judge_loop_37lang.py` | stage16_fullscale | the judge loop at 37 languages | replaced by consensus_loop_37lang |
| `absolute_judge_bonus.py` | cpu_absolute_bonus | un-normalised judge bonus at 37 languages; break-even judge accuracy | |

## Understanding the loop's behaviour (CPU, cached)

| file | was | question |
|---|---|---|
| `consensus_flips_and_oscillators.py` | judge_forensics | which pools did consensus flip from right to wrong, and which oscillate |
| `abstained_pools_have_no_gold.py` | abstention_check | is 0% accuracy on abstained pools a bug? No: the gold answer is not in the pool |
| `abstention_flow_through_laps.py` | abstention_flow | who is vetoed, who recovers, who abstains late |
| `route_abstentions_to_judge.py` | abstention_routing | route abstentions to the judge's top chain (preview only; sample is gold-enriched) |
| `per_language_health_check.py` | verify_scale | convergence, accuracy, veto rate per language before scaling up |
| `trace_one_problem.py` | trace_pipeline | **follows one real problem through 5 languages, step by step. Good for learning.** |
