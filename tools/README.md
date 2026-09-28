# tools

Checks that the repository is what it claims to be. All standard-library
Python; `run_selftests.py` also needs numpy.

| command | proves |
|---|---|
| `python tools/check_provenance.py` | every file in `icm/experiments` equals its `~/phase0` original apart from import lines and output paths; `generation/` is byte-identical to git HEAD of `~/multilingual_icm` |
| `python tools/check_imports.py` | every relative import resolves and every imported name exists; nothing still imports a `stageNN` name |
| `python tools/run_selftests.py --baseline` | every experiment's `--self-test` passes and prints exactly what the original prints |
| `python tools/compare_results.py A.json B.json` | two result files are identical (use after a re-run) |

The first three need `~/phase0` and `~/multilingual_icm` to exist. The method
itself is tested by `tests/`, not here.
