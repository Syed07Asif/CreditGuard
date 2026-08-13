"""Baseline profiles: what a promoted model's training population looked
like, captured once at promotion time so every later drift check has a
fixed reference to compare against -- never "last week", always the ACTIVE
model's own training distribution (CLAUDE.md's leakage/reproducibility
spirit applied to monitoring: a moving baseline would silently absorb drift
instead of detecting it).

Persistence is split two ways, deliberately reusing patterns already
established in this codebase rather than inventing a new one:

  * The full profile (per-numeric-feature mean/std/min/max/decile edges,
    per-categorical-feature frequencies, predicted-probability deciles,
    observed default rate, training window) is written as a JSON artifact
    under `<model_dir>/baselines/<model_id>_baseline.json` -- the same
    "artifact on disk, referenced from the registry" shape Phase 7 already
    uses for the SHAP background sample.
  * A reference sample (a fixed-size random sample of the training frame's
    numeric columns, `config/monitoring.yaml`'s `baseline.reference_sample_size`
    rows) is written alongside it as parquet -- PSI and chi-square only need
    the summary stats above, but a genuine two-sample Kolmogorov-Smirnov
    test (`monitoring/drift.py`) needs an actual second sample to compare
    against, not just a mean/std.
  * A compact pointer + summary is merged onto the model's own
    `model_registry.metrics` JSONB via `registry.update_metrics` -- the
    same mechanism Phase 9's `models.performance.backfill` already
    established for "enrich a registered model's metadata after the fact
    without touching the artifact" (CLAUDE.md hard rule 6 is about the
    trained model artifact, not this). No new database table was added for
    this: `db/schema.sql`'s Phase 1 schema already provisioned
    `drift_reports`/`monitoring_metrics`/`data_quality_issues` for this
    phase but not a baseline table, so the existing JSONB-patch mechanism is
    the natural fit rather than a schema change new to this phase.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from creditguard.config import get_settings
from creditguard.models import registry

DEFAULT_MONITORING_CONFIG_PATH = "config/monitoring.yaml"
DEFAULT_N_DECILES = 10
DEFAULT_REFERENCE_SAMPLE_SIZE = 5000


class BaselineNotFoundError(RuntimeError):
    """Raised when no baseline profile has been built for a model yet."""


def load_monitoring_config(path: str | Path = DEFAULT_MONITORING_CONFIG_PATH) -> dict:
    """Load config/monitoring.yaml into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class NumericFeatureBaseline:
    mean: float
    std: float
    min: float
    max: float
    decile_edges: list[float]


@dataclass
class CategoricalFeatureBaseline:
    frequencies: dict[str, float] = field(default_factory=dict)


@dataclass
class BaselineProfile:
    """Everything `monitoring/drift.py` needs to score current production
    data against a fixed training-time reference.
    """

    model_id: str
    dataset_version: str
    training_window_start: str
    training_window_end: str
    observed_default_rate: float
    n_rows: int
    created_at: str
    numeric_features: dict[str, NumericFeatureBaseline]
    categorical_features: dict[str, CategoricalFeatureBaseline]
    prediction_probability_deciles: list[float]
    sample_path: str


def _decile_edges(values: pd.Series, n_deciles: int) -> list[float]:
    """`n_deciles` bin edges spanning `values`' quantiles, with the outer
    edges widened to +/-inf so a current value outside the training range
    still falls inside the outermost bin instead of being dropped.
    """
    quantiles = np.linspace(0.0, 1.0, n_deciles + 1)
    edges = np.unique(values.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        # Degenerate (constant) feature: a single bin spanning everything.
        edges = np.array([values.min(), values.max()])
        if edges[0] == edges[1]:
            edges = np.array([edges[0] - 1.0, edges[0] + 1.0])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(e) for e in edges]


def build_baseline_profile(
    *,
    model_id: str,
    dataset_version: str,
    train_frame: pd.DataFrame,
    p_train: np.ndarray,
    y_train: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    training_window_start: Any,
    training_window_end: Any,
    n_deciles: int = DEFAULT_N_DECILES,
    reference_sample_size: int = DEFAULT_REFERENCE_SAMPLE_SIZE,
    seed: int = 42,
) -> tuple[BaselineProfile, pd.DataFrame]:
    """Build a `BaselineProfile` (and its accompanying reference sample
    DataFrame) from a model's own training data -- `train_frame` is the
    pre-`ColumnTransformer` frame (human-readable ratio/behavioural
    features, not one-hot/ordinal-encoded), the same shape
    `creditguard.models.train.TrainingData.train_frame` produces.
    """
    numeric: dict[str, NumericFeatureBaseline] = {}
    for column in numeric_columns:
        if column not in train_frame.columns:
            continue
        values = pd.to_numeric(train_frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        numeric[column] = NumericFeatureBaseline(
            mean=float(values.mean()),
            std=float(values.std(ddof=0)),
            min=float(values.min()),
            max=float(values.max()),
            decile_edges=_decile_edges(values, n_deciles),
        )

    categorical: dict[str, CategoricalFeatureBaseline] = {}
    for column in categorical_columns:
        if column not in train_frame.columns:
            continue
        frequencies = train_frame[column].astype(str).value_counts(normalize=True)
        categorical[column] = CategoricalFeatureBaseline(
            frequencies={str(k): float(v) for k, v in frequencies.items()}
        )

    prob_deciles = _decile_edges(pd.Series(np.asarray(p_train, dtype=float)), n_deciles)

    sample_columns = [c for c in numeric_columns if c in train_frame.columns]
    sample = train_frame[sample_columns].copy()
    sample["__p_default__"] = np.asarray(p_train, dtype=float)
    if len(sample) > reference_sample_size:
        sample = sample.sample(n=reference_sample_size, random_state=seed)
    sample = sample.reset_index(drop=True)

    profile = BaselineProfile(
        model_id=model_id,
        dataset_version=dataset_version,
        training_window_start=str(training_window_start),
        training_window_end=str(training_window_end),
        observed_default_rate=float(np.mean(np.asarray(y_train))),
        n_rows=int(len(train_frame)),
        created_at=datetime.now(UTC).isoformat(),
        numeric_features=numeric,
        categorical_features=categorical,
        prediction_probability_deciles=prob_deciles,
        sample_path="",  # filled in by save_baseline once the path is known
    )
    return profile, sample


def _baseline_dir(model_dir: Path | None = None) -> Path:
    settings = get_settings()
    base = model_dir if model_dir is not None else settings.model_dir
    return Path(base) / "baselines"


def _profile_to_dict(profile: BaselineProfile) -> dict[str, Any]:
    payload = asdict(profile)
    return payload


def _profile_from_dict(raw: dict[str, Any]) -> BaselineProfile:
    numeric = {
        name: NumericFeatureBaseline(**values)
        for name, values in raw["numeric_features"].items()
    }
    categorical = {
        name: CategoricalFeatureBaseline(**values)
        for name, values in raw["categorical_features"].items()
    }
    return BaselineProfile(
        model_id=raw["model_id"],
        dataset_version=raw["dataset_version"],
        training_window_start=raw["training_window_start"],
        training_window_end=raw["training_window_end"],
        observed_default_rate=raw["observed_default_rate"],
        n_rows=raw["n_rows"],
        created_at=raw["created_at"],
        numeric_features=numeric,
        categorical_features=categorical,
        prediction_probability_deciles=raw["prediction_probability_deciles"],
        sample_path=raw["sample_path"],
    )


def save_baseline(
    profile: BaselineProfile,
    sample: pd.DataFrame,
    model_dir: Path | None = None,
) -> Path:
    """Persist `profile`'s JSON artifact and `sample`'s parquet artifact,
    and merge a compact pointer/summary onto the model's own
    `model_registry.metrics` row. Returns the JSON artifact's path.
    """
    baseline_dir = _baseline_dir(model_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    sample_path = baseline_dir / f"{profile.model_id}_sample.parquet"
    sample.to_parquet(sample_path, index=False)
    profile.sample_path = str(sample_path)

    json_path = baseline_dir / f"{profile.model_id}_baseline.json"
    json_path.write_text(
        json.dumps(_profile_to_dict(profile), indent=2), encoding="utf-8"
    )

    registry.update_metrics(
        profile.model_id,
        {
            "baseline": {
                "artifact_path": str(json_path),
                "sample_path": str(sample_path),
                "n_rows": profile.n_rows,
                "observed_default_rate": profile.observed_default_rate,
                "training_window_start": profile.training_window_start,
                "training_window_end": profile.training_window_end,
                "created_at": profile.created_at,
            }
        },
    )
    return json_path


def load_baseline(model_id: str, model_dir: Path | None = None) -> BaselineProfile:
    """Load `model_id`'s persisted baseline profile.

    Raises:
        BaselineNotFoundError: if no baseline has been built for this model
            yet (run `python -m creditguard.monitoring.baseline` after
            promotion first).
    """
    json_path = _baseline_dir(model_dir) / f"{model_id}_baseline.json"
    if not json_path.exists():
        raise BaselineNotFoundError(
            f"No baseline profile found for model_id={model_id!r} at "
            f"{json_path} -- run `python -m creditguard.monitoring.baseline "
            "--model-id ...` (or let the pipeline orchestrator's --register "
            "stage do it) after promoting this model."
        )
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return _profile_from_dict(raw)


def load_baseline_sample(profile: BaselineProfile) -> pd.DataFrame:
    """Load the reference sample DataFrame a `BaselineProfile` points to."""
    return pd.read_parquet(profile.sample_path)


def build_and_save_baseline_for_model(
    model_id: str,
    *,
    features_config_path: str | Path = "config/features.yaml",
    cleaning_config_path: str | Path = "config/validation_rules.yaml",
    monitoring_config_path: str | Path = DEFAULT_MONITORING_CONFIG_PATH,
    data_dir: Path | None = None,
) -> Path:
    """Build and persist a baseline profile for an already-registered model,
    from its own training data -- reuses
    `creditguard.models.train.load_training_data` (the same temporal
    train/val/test split and fitted feature pipeline that model's own
    training run used) so the baseline reflects exactly what that model was
    trained on, and `creditguard.models.performance.compute_performance_artifacts`'
    pattern of rebuilding a registered model's own split on demand rather
    than re-reading anything persisted only in-memory during training.

    Called after `creditguard.models.registry.register_model` promotes a
    model -- by the pipeline orchestrator's `--register` stage, or by
    `monitoring.retraining` after a successful challenger promotion.
    """
    import joblib

    from creditguard.db.repository import ModelRegistryRepository
    from creditguard.models.train import load_features_config, load_training_data
    from creditguard.validation.engine import load_rule_config

    model_row = ModelRegistryRepository().get_by_id(model_id)
    if model_row is None:
        raise BaselineNotFoundError(f"No model_registry row for model_id={model_id!r}")

    features_config = load_features_config(features_config_path)
    cleaning_config = load_rule_config(cleaning_config_path)
    monitoring_config = load_monitoring_config(monitoring_config_path)
    baseline_config = monitoring_config["baseline"]

    data = load_training_data(
        model_row["dataset_version"], features_config, cleaning_config, data_dir
    )
    calibrated_model = joblib.load(model_row["artifact_path"])
    p_train = calibrated_model.predict_proba(data.X_train)[:, 1]

    columns = features_config["feature_columns"]
    profile, sample = build_baseline_profile(
        model_id=model_id,
        dataset_version=model_row["dataset_version"],
        train_frame=data.train_frame,
        p_train=p_train,
        y_train=data.y_train,
        numeric_columns=columns["numeric"],
        categorical_columns=columns["categorical"] + columns["ordinal"],
        training_window_start=data.train_frame["application_date"].min(),
        training_window_end=data.train_frame["application_date"].max(),
        n_deciles=baseline_config["n_deciles"],
        reference_sample_size=baseline_config["reference_sample_size"],
        seed=get_settings().seed,
    )
    return save_baseline(profile, sample)


def main(argv: list[str] | None = None) -> None:
    """CLI: build and persist a baseline profile for a model (the active
    model if `--model-id` is omitted).

        python -m creditguard.monitoring.baseline [--model-id ...]
    """
    parser = argparse.ArgumentParser(
        description="Build a CreditGuard Phase 10 baseline profile."
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--features-config", default="config/features.yaml")
    parser.add_argument("--cleaning-config", default="config/validation_rules.yaml")
    parser.add_argument("--monitoring-config", default=DEFAULT_MONITORING_CONFIG_PATH)
    args = parser.parse_args(argv)

    model_id = args.model_id
    if model_id is None:
        model_row = registry.get_active_model()
        if model_row is None:
            raise BaselineNotFoundError(
                "No active model registered -- run `python -m "
                "creditguard.models.train --register-best` first."
            )
        model_id = model_row["model_id"]

    path = build_and_save_baseline_for_model(
        model_id,
        features_config_path=args.features_config,
        cleaning_config_path=args.cleaning_config,
        monitoring_config_path=args.monitoring_config,
    )
    print(f"Built baseline profile for model_id={model_id} -> {path}")


if __name__ == "__main__":
    main()
