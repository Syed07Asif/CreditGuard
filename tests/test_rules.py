"""Unit tests for individual validation rules: exact violation counts on tiny frames."""

from __future__ import annotations

import pandas as pd

from creditguard.validation.rules import (
    CreditUtilizationRule,
    DuplicateRecordRule,
    EmploymentPlausibilityRule,
    ExpenseConsistencyRule,
    ImpossibleAgeRule,
    IncomeConsistencyRule,
    InvalidDateRule,
    MissingValueRule,
    NegativeFinancialRule,
    NumericRangeRule,
    OrphanRecordRule,
    Severity,
    TemporalLeakageRule,
)


def test_missing_value_rule_counts_missing_cells() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "as_of_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "total_assets": [100.0, None, None],
        }
    )
    rule = MissingValueRule("financial_profiles", "total_assets", threshold=0.0)
    violations = rule.check({"financial_profiles": df})
    assert len(violations) == 2


def test_missing_value_rule_below_threshold_reports_nothing() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "as_of_date": ["2024-01-01"] * 4,
            "total_assets": [100.0, 200.0, 300.0, None],
        }
    )
    rule = MissingValueRule("financial_profiles", "total_assets", threshold=0.5)
    violations = rule.check({"financial_profiles": df})
    assert len(violations) == 0


def test_duplicate_record_rule_key_based() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2"],
            "age": [30, 30, 40],
        }
    )
    rule = DuplicateRecordRule("customers", keys=["customer_id"])
    violations = rule.check({"customers": df})
    assert len(violations) == 2
    assert set(violations["record_key"]) == {"C1"}


def test_duplicate_record_rule_exact_row() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "age": [30, 30, 40],
        }
    )
    rule = DuplicateRecordRule("customers", exact=True)
    violations = rule.check({"customers": df})
    assert len(violations) == 0

    df_with_exact_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    violations = rule.check({"customers": df_with_exact_dupe})
    assert len(violations) == 2


def test_numeric_range_rule() -> None:
    df = pd.DataFrame({"customer_id": ["C1", "C2", "C3"], "dependents": [-1, 5, 15]})
    rule = NumericRangeRule("customers", "dependents", min=0, max=10)
    violations = rule.check({"customers": df})
    assert len(violations) == 2


def test_negative_financial_rule() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "annual_income": [-100.0, 200.0],
            "monthly_income": [50.0, -20.0],
        }
    )
    rule = NegativeFinancialRule(
        "customers", columns=["annual_income", "monthly_income"]
    )
    violations = rule.check({"customers": df})
    assert len(violations) == 2


def test_impossible_age_rule() -> None:
    df = pd.DataFrame({"customer_id": ["C1", "C2", "C3"], "age": [10, 30, 150]})
    rule = ImpossibleAgeRule(min=18, max=100)
    violations = rule.check({"customers": df})
    assert len(violations) == 2


def test_invalid_date_rule() -> None:
    df = pd.DataFrame(
        {
            "loan_id": ["L1", "L2", "L3", "L4"],
            "application_date": [
                "2024-01-01",
                "2024-01-01",
                "2099-01-01",
                "2024-01-01",
            ],
            "decision_date": ["2024-01-05", "2023-12-25", "2099-01-05", "not-a-date"],
        }
    )
    rule = InvalidDateRule(
        "loan_applications",
        columns=["application_date", "decision_date"],
        order_pairs=[["application_date", "decision_date"]],
        allow_future=False,
    )
    violations = rule.check({"loan_applications": df})
    # L2: decision before application (1). L3: both dates future (2).
    # L4: unparseable decision date (1).
    assert len(violations) == 4


def test_credit_utilization_rule() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "as_of_date": ["2024-01-01"] * 3,
            "credit_utilization": [0.5, 1.8, 0.3],
            "total_outstanding": [500.0, 900.0, 2000.0],
            "total_credit_limit": [1000.0, 1000.0, 1000.0],
        }
    )
    rule = CreditUtilizationRule(min=0.0, max=1.5, outstanding_over_limit_margin=1.5)
    violations = rule.check({"credit_history": df})
    assert len(violations) == 2


def test_income_consistency_rule() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "annual_income": [120000.0, 120000.0],
            "monthly_income": [10000.0, 5000.0],
        }
    )
    rule = IncomeConsistencyRule(tolerance_fraction=0.1)
    violations = rule.check({"customers": df})
    assert len(violations) == 1


def test_expense_consistency_rule() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "as_of_date": ["2024-01-01"] * 2,
            "monthly_expenses": [3000.0, 5000.0],
            "monthly_emi": [1000.0, 3000.0],
            "monthly_income": [4000.0, 4000.0],
        }
    )
    rule = ExpenseConsistencyRule(income_multiple=1.5)
    violations = rule.check({"financial_profiles": df})
    assert len(violations) == 1


def test_orphan_record_rule() -> None:
    customers = pd.DataFrame({"customer_id": ["C1", "C2"]})
    loans = pd.DataFrame({"loan_id": ["L1", "L2"], "customer_id": ["C1", "C3"]})
    rule = OrphanRecordRule(
        child_table="loan_applications", parent_table="customers", key="customer_id"
    )
    violations = rule.check({"customers": customers, "loan_applications": loans})
    assert len(violations) == 1


def test_employment_plausibility_rule() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "age": [30, 20],
            "employment_years": [10.0, 10.0],
        }
    )
    rule = EmploymentPlausibilityRule(min_working_age=16)
    violations = rule.check({"customers": df})
    assert len(violations) == 1


def test_temporal_leakage_rule() -> None:
    loans = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "customer_id": ["C1", "C1"],
            "application_date": ["2024-01-01", "2024-03-01"],
        }
    )
    snapshots = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "as_of_date": ["2024-02-01", "2024-05-01"],
        }
    )
    rule = TemporalLeakageRule(
        snapshot_table="financial_profiles", loan_table="loan_applications"
    )
    violations = rule.check(
        {"loan_applications": loans, "financial_profiles": snapshots}
    )
    assert len(violations) == 1


def test_rule_severity_is_preserved() -> None:
    rule = ImpossibleAgeRule(severity=Severity.ERROR)
    assert rule.severity == Severity.ERROR


def test_validation_detects_injected_errors_at_scale() -> None:
    """End-to-end: run the real rule config against a generated dataset and check
    detection rate per injected error type against the generator's own manifest.
    """
    from creditguard.data.generator import generate_dataset, load_config
    from creditguard.validation.engine import build_registry, load_rule_config, run

    gen_config = load_config("config/data_generation.yaml")
    dataset = generate_dataset(gen_config, seed=42, n_customers=8000)

    dataframes = {
        "customers": dataset.customers,
        "loan_applications": dataset.loan_applications,
        "financial_profiles": dataset.financial_profiles,
        "credit_history": dataset.credit_history,
        "loan_outcomes": dataset.loan_outcomes,
    }
    rule_config = load_rule_config("config/validation_rules.yaml")
    registry = build_registry(rule_config)
    result = run(dataframes, registry, dataset_version="test")

    manifest = dataset.injection_manifest

    def flagged_keys(rule_name: str) -> set[str]:
        v = result.violations
        return set(v.loc[v["rule_name"] == rule_name, "record_key"])

    checks = {
        "out_of_range_age": (
            "impossible_age",
            manifest["out_of_range_age"]["customer_ids"],
        ),
        "negative_financial_value": (
            "negative_financial:customers",
            manifest["negative_financial_value"]["customer_ids"],
        ),
        "inconsistent_income": (
            "income_consistency",
            manifest["inconsistent_income"]["customer_ids"],
        ),
        "duplicate_customer": (
            "duplicate_record:customers:customer_id",
            manifest["duplicate_customer"]["customer_ids"],
        ),
        "missing_value": (
            "missing_value:financial_profiles.total_assets",
            manifest["missing_value"]["row_keys"],
        ),
        "impossible_utilization": (
            "credit_utilization",
            manifest["impossible_utilization"]["row_keys"],
        ),
    }

    for error_type, (rule_name, expected_keys) in checks.items():
        expected = {str(k) for k in expected_keys}
        detected = flagged_keys(rule_name) & expected
        rate = len(detected) / len(expected) if expected else 1.0
        assert (
            rate >= 0.95
        ), f"{error_type}: only detected {rate:.2%} of injected rows via {rule_name}"
