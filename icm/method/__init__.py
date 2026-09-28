"""The ICM method, in four small files. Read them in this order:

    data.py        read the generated chains; compare answers
    vote.py        commit, veto, and let the other languages vote
    regenerate.py  give a stuck language a second try
    metrics.py     coverage and precision (the only place gold is read)
    run.py         puts them together:  python -m icm.method.run decide

Every function here is copied from the experiment that produced the headline
result (icm/experiments/q04_full_scale_headline/full_scale_37x1500.py and the files it
imports). tests/test_method_matches_experiments.py proves the two give
identical output.
"""
