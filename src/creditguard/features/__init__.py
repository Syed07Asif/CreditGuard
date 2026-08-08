"""Feature engineering: composable scikit-learn transformers with leakage
prevention enforced by code, not just convention.

`leakage.py` is the only module that decides what is/isn't allowed into a
feature frame and how loans get matched to their point-in-time snapshots;
`ratios.py`/`behavioural.py` compute engineered features; `encoders.py`/
`pipeline.py` assemble the full scikit-learn `Pipeline`; `build.py` is the CLI
that splits, fits and writes versioned feature matrices.
"""
