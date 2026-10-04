"""Analyzers that turn code, data and results into findings.

Each sub-package is independent and usable on its own:

``static``       AST analysis of research code (rules QP001–QP015)
``causal``       runtime future-perturbation test (QP-CAUSAL-*)
``leakage``      value-level leakage diagnostics (QP-LEAK-*)
``execution``    execution-realism analysis (QP-EXEC-*)
``statistical``  statistics and selection-bias analysis (QP-STAT-*, QP-VAL-*)
"""
