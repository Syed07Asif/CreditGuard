"""CLI: fit the Phase 4 feature pipeline and write versioned train/val/test matrices.

    python -m creditguard.features.build --dataset-version ds_..._clean \
        --split-strategy temporal --output data/features/<version>/

Splits temporally by `application_date`: train = oldest 70% of loans,
validation = next 15%, test = most recent 15%. The feature pipeline (see
`creditguard.features.pipeline`) is fit on the train split only -- every
learned statistic (imputer medians, scaler mean/std, one-hot categories,
quantile bin edges) comes from train rows alone; val/test are transformed
with those same fitted statistics, never refit.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from creditguard.config import get_settings
from creditguard.data.versioning import read_dataset_tables
from creditguard.features.leakage import assert_no_leakage, check_target_correlation
from creditguard.features.pipeline import build_feature_pipeline
from creditguard.validation.engine import load_rule_config

Tables = dict[str, pd.DataFrame]

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15


def load_features_config(path: str | Path) -> dict[str, Any]:
    """Load config/features.yaml into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def temporal_split(
    loan_applications: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> tuple[set[str], set[str], set[str]]:
    """Split loan_ids into (train, val, test) by application_date, oldest first."""
    ordered = loan_applications.assign(
        application_date=pd.to_datetime(loan_applications["application_date"])
    ).sort_values(["application_date", "loan_id"])
    n = len(ordered)
    train_end = int(round(n * train_fraction))
    val_end = int(round(n * (train_fraction + val_fraction)))
    loan_ids = ordered["loan_id"].to_numpy()
    return (
        set(loan_ids[:train_end]),
        set(loan_ids[train_end:val_end]),
        set(loan_ids[val_end:]),
    )


def filter_tables(tables: Tables, loan_ids: set[str]) -> Tables:
    """Restrict every table to the rows relevant to `loan_ids`."""
    loans = tables["loan_applications"]
    loans = loans[loans["loan_id"].isin(loan_ids)]
    customer_ids = set(loans["customer_id"])
    return {
        "customers": tables["customers"][
            tables["customers"]["customer_id"].isin(customer_ids)
        ].reset_index(drop=True),
        "loan_applications": loans.reset_index(drop=True),
        "financial_profiles": tables["financial_profiles"][
            tables["financial_profiles"]["customer_id"].isin(customer_ids)
        ].reset_index(drop=True),
        "credit_history": tables["credit_history"][
            tables["credit_history"]["customer_id"].isin(customer_ids)
        ].reset_index(drop=True),
        "loan_outcomes": tables["loan_outcomes"][
            tables["loan_outcomes"]["loan_id"].isin(loan_ids)
        ].reset_index(drop=True),
    }


def align_target(merged: pd.DataFrame, loan_outcomes: pd.DataFrame) -> pd.Series:
    """`default_12m` for each row of `merged`, looked up by loan_id (not position)."""
    outcome_by_loan = loan_outcomes.set_index("loan_id")["default_12m"]
    y = merged["loan_id"].map(outcome_by_loan)
    if y.isna().any():
        missing = merged.loc[y.isna(), "loan_id"].tolist()
        raise ValueError(f"No outcome found for loan_id(s): {missing[:10]}")
    return y.rename("default_12m").astype(int).reset_index(drop=True)


def build_features(
    dataset_version: str,
    split_strategy: str,
    output_dir: Path,
    features_config: dict[str, Any],
    cleaning_config: dict[str, Any],
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit the feature pipeline on the train split and write X/y matrices."""
    if split_strategy != "temporal":
        raise ValueError(f"Unsupported split strategy: {split_strategy!r}")

    settings = get_settings()
    data_root = data_dir if data_dir else settings.data_dir / "processed"
    tables = read_dataset_tables(dataset_version, data_root)

    train_ids, val_ids, test_ids = temporal_split(tables["loan_applications"])
    train_tables = filter_tables(tables, train_ids)
    val_tables = filter_tables(tables, val_ids)
    test_tables = filter_tables(tables, test_ids)

    pipeline = build_feature_pipeline(features_config, cleaning_config)
    # Two-phase fit: the "cleaning_and_merge" step changes row granularity
    # (dict-of-tables -> one row per loan), so the target can only be aligned
    # by loan_id *after* it runs. `rest` shares the same step objects as
    # `pipeline` (sklearn Pipeline slicing doesn't clone), so fitting through
    # it leaves `pipeline` itself fully fitted and ready to persist.
    merge_step = pipeline.named_steps["cleaning_and_merge"]
    rest = pipeline[1:]

    train_merged = merge_step.fit_transform(train_tables)
    val_merged = merge_step.transform(val_tables)
    test_merged = merge_step.transform(test_tables)

    y_train = align_target(train_merged, train_tables["loan_outcomes"])
    y_val = align_target(val_merged, val_tables["loan_outcomes"])
    y_test = align_target(test_merged, test_tables["loan_outcomes"])

    X_train = rest.fit_transform(train_merged, y_train)
    X_val = rest.transform(val_merged)
    X_test = rest.transform(test_merged)

    feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert_no_leakage(feature_names)
    correlation_offenders = check_target_correlation(
        pd.DataFrame(X_train, columns=feature_names), y_train
    )

    for name, X in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{name} contains NaN or inf after the feature pipeline")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, X in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
        pd.DataFrame(X, columns=feature_names).to_parquet(
            output_dir / f"{name}.parquet", index=False
        )
    for name, y in (("y_train", y_train), ("y_val", y_val), ("y_test", y_test)):
        y.to_frame().to_parquet(output_dir / f"{name}.parquet", index=False)

    date_ranges = {
        split: {
            "min": str(merged["application_date"].min().date()),
            "max": str(merged["application_date"].max().date()),
            "n_loans": int(len(merged)),
        }
        for split, merged in (
            ("train", train_merged),
            ("val", val_merged),
            ("test", test_merged),
        )
    }

    metadata = {
        "source_dataset_version": dataset_version,
        "output_version": output_dir.name,
        "split_strategy": split_strategy,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "date_ranges": date_ranges,
        "target_correlation_offenders": correlation_offenders,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (output_dir / "features_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    pipeline_path = settings.model_dir / f"feature_pipeline_{output_dir.name}.joblib"
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, pipeline_path)
    feature_names_path = settings.model_dir / f"feature_pipeline_{output_dir.name}.json"
    feature_names_path.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "dtypes": ["float64"] * len(feature_names),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "metadata": metadata,
        "pipeline_path": str(pipeline_path),
        "feature_names_path": str(feature_names_path),
    }


def _print_summary(metadata: dict[str, Any]) -> None:
    print(
        f"Feature build: {metadata['source_dataset_version']} -> "
        f"{metadata['output_version']}"
    )
    print(f"\n{metadata['n_features']} features:")
    for name in metadata["feature_names"]:
        print(f"  {name}")
    print("\nMatrix shapes:")
    for split in ("X_train", "X_val", "X_test"):
        print(f"  {split}: {tuple(metadata['shapes'][split])}")
    print("\nDate ranges:")
    for split in ("train", "val", "test"):
        r = metadata["date_ranges"][split]
        print(f"  {split}: {r['min']} .. {r['max']} ({r['n_loans']} loans)")
    if metadata["target_correlation_offenders"]:
        print(
            "\nWARNING: high target-correlation features (review for leakage): "
            f"{metadata['target_correlation_offenders']}"
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: build and persist versioned feature matrices."""
    parser = argparse.ArgumentParser(description="Build CreditGuard feature matrices.")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--split-strategy", default="temporal", choices=["temporal"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--features-config", default="config/features.yaml")
    parser.add_argument("--cleaning-config", default="config/validation_rules.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    features_config = load_features_config(args.features_config)
    cleaning_config = load_rule_config(args.cleaning_config)
    data_dir = Path(args.data_dir) if args.data_dir else None

    result = build_features(
        dataset_version=args.dataset_version,
        split_strategy=args.split_strategy,
        output_dir=Path(args.output),
        features_config=features_config,
        cleaning_config=cleaning_config,
        data_dir=data_dir,
    )
    _print_summary(result["metadata"])
    print(f"\nSaved pipeline: {result['pipeline_path']}")
    print(f"Saved feature names/dtypes: {result['feature_names_path']}")


if __name__ == "__main__":
    main()
