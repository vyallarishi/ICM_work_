# Glossary

Words used throughout the code and results, in the order you will meet them.

**chain**: one attempt by the model to solve one problem: the reasoning text
plus a final answer (`#### 42`).

**pool**: all chains for one problem in one language (about 20 in the real
data, up to 8 after the *view*). Id: `train_<lang>_<problem>`.

**gold**: the correct answer from the dataset. Read only when measuring
results (`icm/method/metrics.py`), never when deciding.

**share**: the fraction of a pool's chains that give its most common answer.

**plurality**: a pool's most common answer.

**landslide**: a pool with share ≥ 0.70. It commits to its plurality on its
own (unless vetoed).

**contested**: a pool with share < 0.70. The other languages decide its answer.

**world**: for one problem, the other languages' current answers.

**strong world answer**: an answer held by ≥ 70% of the other languages.

**veto**: a landslide pool whose plurality disagrees with a strong world
answer. It loses its answer and is treated as contested.

**abstain**: a contested pool ends with no answer. This happens when the
world's strong answer is one this language never produced.

**generate constraint**: a language can only commit an answer that appears in
its own chains. The world can pick among them, not add to them.

**converge**: the contested pools re-vote 4 times, each time reading the
world's latest answers.

**oscillator**: a contested pool whose answer changed two or more times while
converging. Regeneration only lets it commit if its new chains all agree.

**regeneration**: the second try. A stuck language sees three peer solutions
(two supporting the world's answer, one supporting the largest minority) and
solves the problem 4 more times. New chains count at half weight.

**recovered**: a pool that abstained after voting and got an answer after
regeneration.

**coverage**: pools with an answer ÷ all pools.

**precision**: pools whose answer is correct ÷ pools with an answer.

**tier**: how much training data the model has seen for a language: *high*
(English, German, ...), *mid* (Hindi, Russian, ...), *low* (Amharic, Swahili,
...). Defined in `icm/method/data.py`.

**view / diverse-8**: a pool is cut to at most 8 chains before voting, keeping
one chain per distinct answer first.

**judge**: any attempt to have the model itself say whether a chain is correct.
Eight were tried (`icm/experiments/q02_can_the_model_judge_itself/`); none was good enough.

**AUC**: the standard measure of how well a score separates correct from
incorrect chains: 0.5 is chance, 1.0 is perfect. The bar for a judge was 0.70.

**harvest**: the chains that would become training data: those agreeing with
a committed answer.

**stage NN**: the original numbering of the experiment files in `~/phase0`.
`icm/provenance.json` maps new names to old.
