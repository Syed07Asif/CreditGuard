"""CLI: regenerate every Phase 5 EDA figure and summary table, headlessly.

    python -m creditguard.eda.run_eda --dataset-version ds_..._clean

Reuses Phase 4's own pipeline stages (`CleaningAndMergeStep`, `RatioFeatures`,
`BehaviouralFeatures`) to build one flat, *unscaled* per-loan frame -- the
same logical features `creditguard.features.build` produces, but skipping
its final `ColumnTransformer` (which scales/one-hot-encodes for modelling).
EDA wants raw units (age in years, income in currency, dti as a ratio), not
a StandardScaler's z-scores, so `run_eda` stops one stage earlier.

Where an analysis runs matters and is deliberate, not arbitrary:

  - **Univariate distributions, categorical frequencies and bivariate/band
    default rates run on the full population** (train + val + test
    concatenated) -- these describe the whole loan portfolio, which is also
    what the Phase 9 dashboard needs, not just the fold that will train a
    model.
  - **IV/WOE, the correlation matrix and point-biserial correlations run on
    the train split only** -- these exist to inform Phase 6 feature-selection
    and multicollinearity decisions for whatever the model actually trains
    on, so they follow the same "fit/derive from train only" discipline as
    the rest of this project (see `creditguard.features.leakage`) even
    though nothing here is a fitted statistic that gets persisted.
  - **The temporal check runs on the full population** -- the whole point is
    to see whether the calendar boundary Phase 4 drew (oldest 70% / next
    15% / most recent 15%) lands inside a regime shift, which requires
    seeing every month, not just the training months.

`age_band`/`tenure_band`/`income_band` are produced by `BehaviouralFeatures`
fit on the train split only (identical to Phase 4), then applied to val/test
-- so even though the full population is used for band-level default-rate
charts, the band edges themselves never see val/test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from creditguard.config import get_settings  # noqa: E402
from creditguard.data.versioning import read_dataset_tables  # noqa: E402
from creditguard.eda import bivariate, plots, risk_analysis, univariate  # noqa: E402
from creditguard.features.behavioural import BehaviouralFeatures  # noqa: E402
from creditguard.features.build import (
    align_target,
    filter_tables,
    temporal_split,
)  # noqa: E402
from creditguard.features.pipeline import CleaningAndMergeStep  # noqa: E402
from creditguard.features.ratios import RatioFeatures  # noqa: E402
from creditguard.validation.engine import load_rule_config  # noqa: E402

TARGET_COL = "default_12m"

# The 11 numeric features the Phase 5 brief names explicitly for univariate
# distribution plots (config column names, not the prose names in the brief).
UNIVARIATE_NUMERIC_COLUMNS: tuple[str, ...] = (
    "age",
    "annual_income",
    "monthly_income",
    "loan_amount",
    "interest_rate",
    "loan_tenure_months",
    "dti",
    "post_loan_dti",
    "credit_utilization",
    "savings_to_income",
    "credit_history_years",
)

CATEGORICAL_FREQUENCY_COLUMNS: tuple[str, ...] = (
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "loan_type",
    "loan_purpose",
    "city_tier",
)

# (band_column, category_order) for the required bivariate band breakdowns.
# `credit_history_band` doesn't exist on the merged frame -- it's added by
# `_add_credit_history_band` below, purely for this EDA breakdown (Phase 4's
# engineered features already cover age/income/tenure/utilization bands).
BAND_BREAKDOWNS: tuple[tuple[str, list[str] | None], ...] = (
    ("age_band", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
    ("income_band", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
    ("employment_type", None),
    ("loan_type", None),
    ("credit_history_band", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
    ("utilization_band", ["0-30", "30-50", "50-70", "70-90", "90+"]),
    ("dependents", None),
    ("city_tier", None),
)


def load_features_config(path: str | Path) -> dict[str, Any]:
    """Load config/features.yaml into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _add_credit_history_band(df: pd.DataFrame) -> pd.DataFrame:
    """EDA-only quintile band of `credit_history_years`, for the "credit
    history band" bivariate breakdown the brief asks for.
    """
    df = df.copy()
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    binned = pd.qcut(df["credit_history_years"], q=5, labels=labels, duplicates="drop")
    df["credit_history_band"] = binned.astype(str)
    return df


def build_eda_frame(
    dataset_version: str,
    features_config: dict[str, Any],
    cleaning_config: dict[str, Any],
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """One flat, unscaled per-loan frame covering train+val+test, with
    `default_12m` and a `split` column attached.

    Mirrors `creditguard.features.build.build_features`'s fit discipline:
    `CleaningAndMergeStep`/`BehaviouralFeatures` are fit on the train split
    only, then used to transform val/test -- but stops before the final
    `ColumnTransformer` so every column stays in its original, human-readable
    unit.
    """
    settings = get_settings()
    data_root = data_dir if data_dir else settings.data_dir / "processed"
    tables = read_dataset_tables(dataset_version, data_root)

    train_ids, val_ids, test_ids = temporal_split(tables["loan_applications"])
    train_tables = filter_tables(tables, train_ids)
    val_tables = filter_tables(tables, val_ids)
    test_tables = filter_tables(tables, test_ids)

    merge_step = CleaningAndMergeStep(cleaning_config)
    ratios = RatioFeatures()
    n_bins = features_config.get("behavioural", {}).get("n_quantile_bins", 5)
    behavioural = BehaviouralFeatures(n_bins=n_bins)

    train_merged = merge_step.fit_transform(train_tables)
    val_merged = merge_step.transform(val_tables)
    test_merged = merge_step.transform(test_tables)

    y_train = align_target(train_merged, train_tables["loan_outcomes"])
    y_val = align_target(val_merged, val_tables["loan_outcomes"])
    y_test = align_target(test_merged, test_tables["loan_outcomes"])

    behavioural.fit(train_merged)

    frames = []
    for split_name, merged, y in (
        ("train", train_merged, y_train),
        ("val", val_merged, y_val),
        ("test", test_merged, y_test),
    ):
        frame = ratios.transform(merged)
        frame = behavioural.transform(frame)
        frame = frame.reset_index(drop=True)
        frame[TARGET_COL] = y.to_numpy()
        frame["split"] = split_name
        frames.append(frame)

    full = pd.concat(frames, ignore_index=True)
    return _add_credit_history_band(full)


def run_eda(
    dataset_version: str,
    output_dir: Path,
    tables_dir: Path,
    features_config: dict[str, Any],
    cleaning_config: dict[str, Any],
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Run every Phase 5 analysis and write figures/tables to disk. Returns a
    summary dict (figure count, IV table, high-correlation pairs, regime
    shift verdict) for the caller to print or assert on.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    full = build_eda_frame(dataset_version, features_config, cleaning_config, data_dir)
    train = full[full["split"] == "train"].reset_index(drop=True)

    numeric_columns = features_config["feature_columns"]["numeric"]
    categorical_columns = features_config["feature_columns"]["categorical"]
    ordinal_columns = features_config["feature_columns"]["ordinal"]

    n_figures = 0

    # -- Univariate -----------------------------------------------------
    plots.plot_class_balance(
        full[TARGET_COL], save_path=output_dir / "univariate_class_balance.png"
    )
    n_figures += 1

    numeric_summary = univariate.numeric_summary(full, list(UNIVARIATE_NUMERIC_COLUMNS))
    numeric_summary.to_csv(tables_dir / "univariate_numeric_summary.csv")
    for column in UNIVARIATE_NUMERIC_COLUMNS:
        plots.plot_numeric_distribution(
            full, column, save_path=output_dir / f"univariate_dist_{column}.png"
        )
        n_figures += 1

    freq_tables = univariate.categorical_frequency(
        full, list(CATEGORICAL_FREQUENCY_COLUMNS)
    )
    for column, table in freq_tables.items():
        table.to_csv(tables_dir / f"univariate_freq_{column}.csv")
        plots.plot_categorical_frequency(
            full, column, save_path=output_dir / f"univariate_freq_{column}.png"
        )
        n_figures += 1

    # -- Bivariate / risk: default rate by decile, every numeric driver ---
    decile_tables = []
    for column in numeric_columns:
        decile_df = bivariate.default_rate_by_decile(full, column, TARGET_COL)
        decile_tables.append(decile_df)
        plots.plot_default_rate_by_decile(
            decile_df, save_path=output_dir / f"decile_{column}.png"
        )
        n_figures += 1
    pd.concat(decile_tables, ignore_index=True).to_csv(
        tables_dir / "bivariate_deciles.csv", index=False
    )

    # -- Bivariate / risk: default rate by band ---------------------------
    band_tables = []
    for band_col, order in BAND_BREAKDOWNS:
        band_df = bivariate.default_rate_by_band(full, band_col, TARGET_COL, order)
        band_tables.append(
            band_df.assign(band_column=band_col).rename(
                columns={band_col: "band_value"}
            )
        )
        plots.plot_default_rate_by_band(
            band_df, band_col, save_path=output_dir / f"band_{band_col}.png"
        )
        n_figures += 1
    pd.concat(band_tables, ignore_index=True).to_csv(
        tables_dir / "bivariate_bands.csv", index=False
    )

    # -- IV / WOE (train only) --------------------------------------------
    iv_df = risk_analysis.iv_table(
        train,
        numeric_columns,
        list(categorical_columns) + list(ordinal_columns),
        TARGET_COL,
    )
    iv_df.to_csv(tables_dir / "iv_table.csv", index=False)
    plots.plot_iv_table(iv_df, save_path=output_dir / "iv_table.png")
    n_figures += 1

    # -- Correlation + point-biserial (train only) -------------------------
    corr = bivariate.correlation_matrix(train, numeric_columns)
    corr.to_csv(tables_dir / "correlation_matrix.csv")
    high_corr = bivariate.high_correlation_pairs(corr, threshold=0.8)
    (tables_dir / "high_correlation_pairs.json").write_text(
        json.dumps(high_corr, indent=2), encoding="utf-8"
    )
    plots.plot_correlation_heatmap(
        corr, save_path=output_dir / "correlation_heatmap.png"
    )
    n_figures += 1

    point_biserial = bivariate.point_biserial_correlations(
        train, numeric_columns, TARGET_COL
    )
    point_biserial.to_csv(tables_dir / "point_biserial.csv", index=False)

    # -- Temporal (full population) ----------------------------------------
    monthly = risk_analysis.monthly_volume_and_default_rate(full)
    monthly.to_csv(tables_dir / "temporal_monthly.csv", index=False)
    regime = risk_analysis.detect_regime_shift(monthly)
    (tables_dir / "temporal_regime_shift.json").write_text(
        json.dumps(regime, indent=2), encoding="utf-8"
    )
    plots.plot_temporal_trend(monthly, save_path=output_dir / "temporal_trend.png")
    n_figures += 1

    return {
        "dataset_version": dataset_version,
        "n_rows": int(len(full)),
        "n_figures": n_figures,
        "default_rate_summary": univariate.default_rate_summary(full[TARGET_COL]),
        "iv_table_head": iv_df.head(10).to_dict(orient="records"),
        "high_correlation_pairs": high_corr,
        "regime_shift": regime,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: regenerate every EDA figure and summary table."""
    parser = argparse.ArgumentParser(description="Run CreditGuard Phase 5 EDA.")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output", default="reports/figures/eda")
    parser.add_argument("--tables-output", default="reports/eda/tables")
    parser.add_argument("--features-config", default="config/features.yaml")
    parser.add_argument("--cleaning-config", default="config/validation_rules.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    features_config = load_features_config(args.features_config)
    cleaning_config = load_rule_config(args.cleaning_config)
    data_dir = Path(args.data_dir) if args.data_dir else None

    summary = run_eda(
        dataset_version=args.dataset_version,
        output_dir=Path(args.output),
        tables_dir=Path(args.tables_output),
        features_config=features_config,
        cleaning_config=cleaning_config,
        data_dir=data_dir,
    )

    print(f"EDA run: {summary['dataset_version']} ({summary['n_rows']:,} loans)")
    print(f"Figures written: {summary['n_figures']}")
    print(f"Default rate: {summary['default_rate_summary']}")
    print(f"Top-10 IV features: {summary['iv_table_head']}")
    print(f"High-correlation pairs (|r|>0.8): {summary['high_correlation_pairs']}")
    print(f"Regime shift check: {summary['regime_shift']}")


if __name__ == "__main__":
    main()
