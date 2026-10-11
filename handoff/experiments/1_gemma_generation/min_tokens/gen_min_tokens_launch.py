"""Run the generation repo's generate_chains.py UNCHANGED except that every LLM.generate call samples
with min_tokens=$MIN_TOKENS: end-of-text (and stop tokens) are suppressed for the first N tokens.
Why: without it Gemma 3 12B ends ~89% of its answers immediately (an empty chain); with N=8, 0%.
The SamplingParams class is left alone; each call's params are cloned and min_tokens is set on the clone.
usage: MIN_TOKENS=8 python gen_min_tokens_launch.py <generate_chains.py arguments>"""
import os, runpy, sys
from vllm import LLM
GEN = os.path.join(os.environ["BUNDLE"], "code", "generation", "scripts")
N = int(os.environ["MIN_TOKENS"])
_generate = LLM.generate
_said = []


def generate(self, prompts, sampling_params=None, *args, **kwargs):
    if sampling_params is not None:
        sampling_params = sampling_params.clone()
        sampling_params.min_tokens = N
        if not _said:
            print(f"[gen_min_tokens_launch] sampling with {sampling_params}", flush=True); _said.append(1)
    return _generate(self, prompts, sampling_params, *args, **kwargs)


LLM.generate = generate
print(f"[gen_min_tokens_launch] min_tokens={N}", flush=True)
sys.path.insert(0, GEN)
sys.argv = [os.path.join(GEN, "generate_chains.py")] + sys.argv[1:]
runpy.run_path(os.path.join(GEN, "generate_chains.py"), run_name="__main__")
