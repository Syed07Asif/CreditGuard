"""Model training, imbalance handling, evaluation, calibration, threshold
optimisation, tracking and registration for CreditGuard Phase 6.

`base.py` defines `BaseCreditModel`, the interface `logistic.py`/
`random_forest.py`/`xgboost_model.py` each implement so `train.py` and (from
Phase 8 on) the API never branch on model type. `evaluate.py` is the single
metric suite every other module reports through -- CV, per-segment, and
imbalance-strategy comparisons alike -- so a metric is defined exactly once.
`imbalance.py`, `calibration.py` and `threshold.py` are independent stages
applied in that order to the winning model; `tracking.py` (MLflow) and
`registry.py` (the `model_registry` DB table) both observe the process
without feeding back into it. `train.py` is the CLI that ties all of it
together against a real dataset version.
"""
