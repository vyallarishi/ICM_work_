"""The generation repo's run_experiment.py, unchanged, with its generate_chains.py subprocesses routed
through gen_min_tokens_launch.py (python_mintok stands in for sys.executable)."""
import os, runpy, sys
assert os.environ.get("MIN_TOKENS"), "set MIN_TOKENS"
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["REAL_PYTHON"] = sys.executable
sys.executable = os.path.join(HERE, "python_mintok")
GEN = os.path.join(os.environ["BUNDLE"], "code", "generation", "scripts")
sys.path.insert(0, GEN)
sys.argv = [os.path.join(GEN, "run_experiment.py")] + sys.argv[1:]
runpy.run_path(os.path.join(GEN, "run_experiment.py"), run_name="__main__")
