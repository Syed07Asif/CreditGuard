"""End-to-end test for creditguard.eda.run_eda: builds a small synthetic
dataset version on disk, runs the full headless EDA pipeline against it, and
checks it produces the expected number of figures/tables without error.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from creditguard.data.versioning import TABLE_FILENAMES
from creditguard.eda.run_eda import (
    BAND_BREAKDOWNS,
    CATEGORICAL_FREQUENCY_COLUMNS,
    UNIVARIATE_NUMERIC_COLUMNS,
    load_features_config,
    run_eda,
)
from creditguard.validation.engine import load_rule_config


def _rich_multitable_fixture(n_customers: int = 30) -> dict[str, pd.DataFrame]:
    """Every numeric field varies per row (no zero-variance columns), while
    staying inside every Phase 3 validation bound so `DataCleaner` drops
    nothing -- unlike `tests/test_feature_pipeline.py`'s fixture, which
    intentionally holds several fields constant for a different purpose.
    """
    customer_ids = [f"C{i:03d}" for i in range(n_customers)]
    loan_ids = [f"L{i:03d}" for i in range(n_customers)]

    ages = [22 + i for i in range(n_customers)]
    employment_years = [1.0 + 0.5 * i for i in range(n_customers)]
    annual_income = [300000.0 + 15000.0 * i for i in range(n_customers)]
    monthly_income = [a / 12 for a in annual_income]

    genders = ["MALE", "FEMALE"]
    marital_statuses = ["SINGLE", "MARRIED", "DIVORCED"]
    educations = ["GRADUATE", "POSTGRADUATE", "HIGH_SCHOOL"]
    employment_types = ["SALARIED", "SELF_EMPLOYED", "BUSINESS_OWNER"]
    loan_types = ["PERSONAL", "AUTO", "HOME"]
    loan_purposes = ["DEBT_CONSOLIDATION", "MEDICAL", "WEDDING", "OTHER"]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": ages,
            "gender": [genders[i % len(genders)] for i in range(n_customers)],
            "marital_status": [
                marital_statuses[i % len(marital_statuses)] for i in range(n_customers)
            ],
            "dependents": [i % 4 for i in range(n_customers)],
            "education": [educations[i % len(educations)] for i in range(n_customers)],
            "employment_type": [
                employment_types[i % len(employment_types)] for i in range(n_customers)
            ],
            "employment_years": employment_years,
            "annual_income": annual_income,
            "monthly_income": monthly_income,
            "city_tier": [(i % 3) + 1 for i in range(n_customers)],
        }
    )

    application_dates = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * i)
        for i in range(n_customers)
    ]
    loan_applications = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "customer_id": customer_ids,
            "loan_type": [loan_types[i % len(loan_types)] for i in range(n_customers)],
            "loan_amount": [80000.0 + 5000.0 * i for i in range(n_customers)],
            "loan_tenure_months": [12 + (i % 5) * 6 for i in range(n_customers)],
            "interest_rate": [8.0 + 0.2 * i for i in range(n_customers)],
            "loan_purpose": [
                loan_purposes[i % len(loan_purposes)] for i in range(n_customers)
            ],
            "application_date": application_dates,
            "decision_date": [d + pd.Timedelta(days=3) for d in application_dates],
            "status": ["APPROVED"] * n_customers,
        }
    )

    as_of_dates = [d - pd.Timedelta(days=5) for d in application_dates]
    financial_profiles = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "as_of_date": as_of_dates,
            "monthly_income": monthly_income,
            "monthly_expenses": [
                m * (0.25 + 0.01 * (i % 5)) for i, m in enumerate(monthly_income)
            ],
            "existing_loan_count": [i % 3 for i in range(n_customers)],
            "existing_loan_amount": [1000.0 * (i % 10) for i in range(n_customers)],
            "monthly_emi": [
                m * (0.05 + 0.01 * (i % 4)) for i, m in enumerate(monthly_income)
            ],
            "savings_balance": [m * (2 + i % 6) for i, m in enumerate(monthly_income)],
            "total_assets": [
                a * (0.4 + 0.01 * (i % 10)) for i, a in enumerate(annual_income)
            ],
            "total_liabilities": [5000.0 * (i % 7 + 1) for i in range(n_customers)],
        }
    )
    credit_history = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "as_of_date": as_of_dates,
            "credit_history_months": [12 + 6 * i for i in range(n_customers)],
            "num_credit_accounts": [1 + i % 6 for i in range(n_customers)],
            "total_credit_limit": [100000.0 + 10000.0 * i for i in range(n_customers)],
            "total_outstanding": [
                5000.0 + 2500.0 * (i % 9) for i in range(n_customers)
            ],
            "credit_utilization": [0.1 + 0.02 * (i % 8) for i in range(n_customers)],
            "previous_defaults": [i % 3 for i in range(n_customers)],
            "late_payments_12m": [i % 4 for i in range(n_customers)],
            "missed_payments_12m": [i % 3 for i in range(n_customers)],
            "active_loans": [1 + i % 3 for i in range(n_customers)],
            "closed_loans": [i % 4 for i in range(n_customers)],
        }
    )
    loan_outcomes = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "default_12m": [1 if i % 3 == 0 else 0 for i in range(n_customers)],
            "outcome_observed_date": [
                d + pd.DateOffset(months=12) for d in application_dates
            ],
        }
    )
    return {
        "customers": customers,
        "loan_applications": loan_applications,
        "financial_profiles": financial_profiles,
        "credit_history": credit_history,
        "loan_outcomes": loan_outcomes,
    }


def _write_dataset(
    tables: dict[str, pd.DataFrame], data_root: Path, version: str
) -> None:
    out_dir = data_root / version
    out_dir.mkdir(parents=True, exist_ok=True)
    for table_name, filename in TABLE_FILENAMES.items():
        tables[table_name].to_parquet(out_dir / f"{filename}.parquet", index=False)


def test_run_eda_executes_headlessly_and_produces_expected_figure_count(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "processed"
    version = "ds_test_eda"
    _write_dataset(_rich_multitable_fixture(30), data_root, version)

    features_config = load_features_config("config/features.yaml")
    cleaning_config = load_rule_config("config/validation_rules.yaml")
    numeric_columns = features_config["feature_columns"]["numeric"]

    output_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"

    summary = run_eda(
        dataset_version=version,
        output_dir=output_dir,
        tables_dir=tables_dir,
        features_config=features_config,
        cleaning_config=cleaning_config,
        data_dir=data_root,
    )

    expected_figures = (
        1  # class balance
        + len(UNIVARIATE_NUMERIC_COLUMNS)
        + len(CATEGORICAL_FREQUENCY_COLUMNS)
        + len(numeric_columns)  # one decile chart per numeric driver
        + len(BAND_BREAKDOWNS)
        + 1  # IV table
        + 1  # correlation heatmap
        + 1  # temporal trend
    )
    assert summary["n_figures"] == expected_figures

    png_files = list(output_dir.glob("*.png"))
    assert len(png_files) == expected_figures
    assert (tables_dir / "iv_table.csv").exists()
    assert (tables_dir / "temporal_monthly.csv").exists()
    assert (tables_dir / "correlation_matrix.csv").exists()


def test_run_eda_default_rate_summary_matches_loan_outcomes(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    version = "ds_test_eda_2"
    tables = _rich_multitable_fixture(30)
    _write_dataset(tables, data_root, version)

    features_config = load_features_config("config/features.yaml")
    cleaning_config = load_rule_config("config/validation_rules.yaml")

    summary = run_eda(
        dataset_version=version,
        output_dir=tmp_path / "figures2",
        tables_dir=tmp_path / "tables2",
        features_config=features_config,
        cleaning_config=cleaning_config,
        data_dir=data_root,
    )

    assert summary["n_rows"] == 30
    expected_n_default = int(tables["loan_outcomes"]["default_12m"].sum())
    assert summary["default_rate_summary"]["n_default"] == expected_n_default
    assert summary["regime_shift"]["regime_shift_detected"] in (True, False)
