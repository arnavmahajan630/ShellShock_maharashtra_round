"""The C interpreter for the frozen subset (package A1, plans/03 §2).

    preprocess.py  text clean-up before parsing (comments, #include, #define, smart quotes)
    values.py      values, literals, printf formatting
    interp.py      Program: parses one learner file and runs one call of one function
    harness.py     run_tests(problem, code, sample_only) and trace(problem, code, test_index)

Callers use ml.runner, or ml.c_interp.harness directly.
"""
