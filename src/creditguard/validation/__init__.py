"""Data validation and cleaning: rule-based quality checks, separate from repair.

`rules.py` / `engine.py` only observe data and report violations. `cleaning.py`
is the only module that transforms data. `report.py` renders the results and
`cli.py` wires both into `python -m creditguard.validation.cli`.
"""
