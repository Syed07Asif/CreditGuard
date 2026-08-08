"""Dataset versioning: naming, and writing/reading generated CreditGuard datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

TABLE_FILENAMES = {
    "customers": "customers",
    "loan_applications": "loans",
    "financial_profiles": "financials",
    "credit_history": "credit",
    "loan_outcomes": "outcomes",
}


def compute_config_hash(config: dict) -> str:
    """Return a short, stable hash of a config dict for use in dataset version names."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def make_dataset_version(config: dict, generated_at: datetime | None = None) -> str:
    """Build a dataset_version string: ds_<YYYYMMDD>_<8-char config hash>."""
    generated_at = generated_at or datetime.now(UTC)
    return f"ds_{generated_at:%Y%m%d}_{compute_config_hash(config)}"


@dataclass
class GeneratedDataset:
    """The five generated tables plus generation metadata and the injection manifest."""

    customers: pd.DataFrame
    loan_applications: pd.DataFrame
    financial_profiles: pd.DataFrame
    credit_history: pd.DataFrame
    loan_outcomes: pd.DataFrame
    metadata: dict
    injection_manifest: dict


def write_dataset(
    dataset: GeneratedDataset, dataset_version: str, output_root: Path
) -> Path:
    """Write a generated dataset's tables, metadata and injection manifest to disk."""
    out_dir = output_root / dataset_version
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "customers": dataset.customers,
        "loan_applications": dataset.loan_applications,
        "financial_profiles": dataset.financial_profiles,
        "credit_history": dataset.credit_history,
        "loan_outcomes": dataset.loan_outcomes,
    }
    for table_name, df in tables.items():
        filename = TABLE_FILENAMES[table_name]
        df.to_parquet(out_dir / f"{filename}.parquet", index=False)

    metadata = dict(dataset.metadata)
    metadata["dataset_version"] = dataset_version
    metadata["row_counts"] = {name: len(df) for name, df in tables.items()}
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "injection_manifest.json").write_text(
        json.dumps(dataset.injection_manifest, indent=2, default=str), encoding="utf-8"
    )
    return out_dir


def read_dataset_tables(
    dataset_version: str, data_root: Path
) -> dict[str, pd.DataFrame]:
    """Read a dataset version's parquet tables back into DataFrames, keyed by name."""
    in_dir = data_root / dataset_version
    return {
        table_name: pd.read_parquet(in_dir / f"{filename}.parquet")
        for table_name, filename in TABLE_FILENAMES.items()
    }


def read_metadata(dataset_version: str, data_root: Path) -> dict:
    """Read a dataset version's metadata.json."""
    return json.loads(
        (data_root / dataset_version / "metadata.json").read_text(encoding="utf-8")
    )


def read_injection_manifest(dataset_version: str, data_root: Path) -> dict:
    """Read a dataset version's injection_manifest.json."""
    path = data_root / dataset_version / "injection_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))
