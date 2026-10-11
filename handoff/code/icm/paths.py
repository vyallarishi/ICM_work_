"""Where analysis scripts read caches and write outputs.

In ~/phase0 every script used `OUT_DIR = Path(__file__).parent`, i.e. the
phase0 directory itself. Here that single directory is ~/multilingual_icm/data/ICM_work_/runs
(outside git). Override with the ICM_RUN_DIR environment variable.
"""
import os
from pathlib import Path

RUN_DIR = Path(os.environ.get("ICM_RUN_DIR", Path(__file__).resolve().parents[2] / "results" / "run_dir"))   # (handoff copy)
RUN_DIR.mkdir(parents=True, exist_ok=True)
