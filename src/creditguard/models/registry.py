"""The `model_registry` table: one row per promoted model version. Exactly
one row is ever `is_active = true` -- `register_model` deactivates whatever
was previously active in the same transaction as inserting the new row, so
there's never a moment with zero or two active models visible to a
concurrent reader. `get_active_model` is what Phases 7-9 call to find which
model to load/serve.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select, update

from creditguard.db.engine import get_session
from creditguard.db.models import ModelRegistry


def _row_to_dict(row: ModelRegistry) -> dict[str, Any]:
    return {
        "model_id": row.model_id,
        "model_version": row.model_version,
        "algorithm": row.algorithm,
        "training_date": row.training_date,
        "dataset_version": row.dataset_version,
        "feature_list": row.feature_list,
        "hyperparameters": row.hyperparameters,
        "metrics": row.metrics,
        "mlflow_run_id": row.mlflow_run_id,
        "artifact_path": row.artifact_path,
        "is_active": row.is_active,
    }


def _next_model_version(existing_versions: list[str]) -> str:
    """Semantic version, minor-bumped from the highest existing version.
    `'1.0.0'` if the registry is empty. Never reuses or overwrites a prior
    version -- each promotion gets a strictly higher one.
    """

    def _parse(version: str) -> tuple[int, int, int]:
        try:
            major, minor, patch = (int(part) for part in version.split("."))
            return (major, minor, patch)
        except (ValueError, AttributeError):
            return (0, 0, 0)

    if not existing_versions:
        return "1.0.0"
    major, minor, _patch = max(_parse(v) for v in existing_versions)
    return f"{major}.{minor + 1}.0"


def register_model(
    *,
    algorithm: str,
    training_date: datetime,
    dataset_version: str,
    feature_list: list[str],
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    mlflow_run_id: str,
    artifact_path: str,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Insert a new `model_registry` row as the active model, deactivating
    whatever was previously active -- both in one transaction, so promotion
    is atomic. Returns the newly-registered row as a dict.
    """
    generated_id = (
        model_id or f"{algorithm}_{training_date:%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"
    )
    with get_session() as session:
        session.execute(
            update(ModelRegistry)
            .where(ModelRegistry.is_active.is_(True))
            .values(is_active=False)
        )
        existing_versions = list(
            session.execute(select(ModelRegistry.model_version)).scalars().all()
        )
        row = ModelRegistry(
            model_id=generated_id,
            model_version=_next_model_version(existing_versions),
            algorithm=algorithm,
            training_date=training_date,
            dataset_version=dataset_version,
            feature_list=feature_list,
            hyperparameters=hyperparameters,
            metrics=metrics,
            mlflow_run_id=mlflow_run_id,
            artifact_path=artifact_path,
            is_active=True,
        )
        session.add(row)
        session.flush()
        return _row_to_dict(row)


def get_active_model() -> dict[str, Any] | None:
    """The currently active `model_registry` row, or `None` if none is
    active. Used by Phases 7-9 to find which model to load/serve.
    """
    with get_session() as session:
        row = session.execute(
            select(ModelRegistry).where(ModelRegistry.is_active.is_(True))
        ).scalar_one_or_none()
        return _row_to_dict(row) if row is not None else None
