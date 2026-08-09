# Imbalance strategy comparison

Same model family, same train/validation split, one imbalance strategy varied at a time. Resampling strategies (SMOTE, random undersampling, SMOTE+Tomek) are applied to the training fold only, via an `imblearn.pipeline.Pipeline` -- never to the validation data these metrics are computed on.

## logistic_regression

| Strategy | PR-AUC | ROC-AUC | Recall@P>=0.50 | Brier | Calibration slope |
|---|---|---|---|---|---|
| none | 0.547 | 0.874 | 0.554 | 0.0680 | 1.040 |
| class_weight | 0.546 | 0.874 | 0.559 | 0.1393 | 1.010 |
| smote | 0.544 | 0.873 | 0.563 | 0.1378 | 0.937 |
| random_undersampling | 0.546 | 0.874 | 0.554 | 0.1408 | 1.093 |
| smote_tomek | 0.544 | 0.873 | 0.563 | 0.1378 | 0.937 |

## xgboost

| Strategy | PR-AUC | ROC-AUC | Recall@P>=0.50 | Brier | Calibration slope |
|---|---|---|---|---|---|
| none | 0.539 | 0.873 | 0.529 | 0.0686 | 0.988 |
| class_weight | 0.539 | 0.871 | 0.535 | 0.1369 | 0.993 |
| smote | 0.524 | 0.869 | 0.522 | 0.0709 | 1.109 |
| random_undersampling | 0.528 | 0.871 | 0.529 | 0.1423 | 0.970 |
| smote_tomek | 0.523 | 0.868 | 0.524 | 0.0711 | 1.109 |

## random_forest

| Strategy | PR-AUC | ROC-AUC | Recall@P>=0.50 | Brier | Calibration slope |
|---|---|---|---|---|---|
| none | 0.522 | 0.864 | 0.529 | 0.0700 | 1.020 |
| class_weight | 0.518 | 0.864 | 0.508 | 0.1074 | 1.017 |
| smote | 0.492 | 0.859 | 0.461 | 0.0808 | 1.095 |
| random_undersampling | 0.504 | 0.863 | 0.500 | 0.1483 | 1.062 |
| smote_tomek | 0.489 | 0.859 | 0.461 | 0.0810 | 1.087 |

## Reading this table

Resampling (SMOTE / undersampling / SMOTE+Tomek) typically raises recall at a fixed precision relative to `none`, because it shifts the decision boundary toward the minority class -- but it does so by training on a class distribution that no longer matches reality, which is exactly what breaks calibration: the resulting probabilities are shifted toward the resampled (roughly 50/50) rate rather than the true ~11% default rate, so calibration slope and Brier score are typically worse for the resampling strategies than for `none` or `class_weight`, even when PR-AUC/recall improve. **If a resampling strategy is used in production, its probabilities must be recalibrated** (see `calibration.py`) before they're treated as default probabilities anywhere downstream (Phase 7's score conversion assumes a calibrated probability).

`class_weight` reweights the existing data rather than fabricating or discarding rows, so it changes the loss function's emphasis without changing what the model is actually trained on -- it usually recovers most of resampling's recall gain with much less calibration damage. Recommendation: prefer `class_weight` (`class_weight="balanced"` / XGBoost's `scale_pos_weight`) over resampling for the model that actually gets registered, consistent with `reports/eda/findings.md`'s Phase 6 decision -- this dataset's ~11% default rate is a moderate imbalance, not the kind of extreme (<1%) imbalance where synthetic oversampling earns back its calibration cost.