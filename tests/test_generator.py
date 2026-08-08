"""Tests for creditguard.data.generator: reproducibility, label quality, invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditguard.data.generator import generate_dataset, load_config

N_CUSTOMERS = 6000


@pytest.fixture(scope="module")
def config() -> dict:
    """Load the real data-generation config used for the full-scale generator run."""
    return load_config("config/data_generation.yaml")


@pytest.fixture(scope="module")
def dataset(config: dict):
    """A moderately-sized generated dataset, shared across the statistical checks."""
    return generate_dataset(config, seed=42, n_customers=N_CUSTOMERS)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC; >0.5 means higher `score` predicts y=1."""
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    sum_ranks_pos = ranks[y == 1].sum()
    return (sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def _dti(financial_profiles: pd.DataFrame) -> np.ndarray:
    fin = financial_profiles
    return (
        (fin["monthly_expenses"] + fin["monthly_emi"])
        / fin["monthly_income"].clip(lower=1)
    ).to_numpy()


def _loan_level_income(dataset) -> np.ndarray:
    merged = dataset.loan_applications.merge(
        dataset.customers[["customer_id", "annual_income"]].drop_duplicates(
            "customer_id"
        ),
        on="customer_id",
    )
    return merged["annual_income"].to_numpy()


def test_same_seed_produces_byte_identical_output(config: dict) -> None:
    """Generating twice with the same seed must yield identical tables."""
    ds1 = generate_dataset(config, seed=42, n_customers=500)
    ds2 = generate_dataset(config, seed=42, n_customers=500)
    pd.testing.assert_frame_equal(ds1.customers, ds2.customers)
    pd.testing.assert_frame_equal(ds1.loan_applications, ds2.loan_applications)
    pd.testing.assert_frame_equal(ds1.financial_profiles, ds2.financial_profiles)
    pd.testing.assert_frame_equal(ds1.credit_history, ds2.credit_history)
    pd.testing.assert_frame_equal(ds1.loan_outcomes, ds2.loan_outcomes)


def test_different_seed_produces_different_output(config: dict) -> None:
    """A different seed must not reproduce the same customers table."""
    ds1 = generate_dataset(config, seed=42, n_customers=500)
    ds2 = generate_dataset(config, seed=43, n_customers=500)
    assert not ds1.customers.equals(ds2.customers)


def test_default_rate_within_target_band(dataset, config: dict) -> None:
    """The realised default rate must land inside the configured target band."""
    rate = dataset.metadata["achieved_default_rate"]
    label_cfg = config["label"]
    assert (
        label_cfg["target_default_rate_min"]
        <= rate
        <= label_cfg["target_default_rate_max"]
    )


def test_snapshot_dates_never_after_application_date(dataset) -> None:
    """No financial/credit snapshot may be dated after the application it supports.

    financial_profiles, credit_history, loan_applications and loan_outcomes are all
    built from one shared per-loan frame and never independently reordered, so row i
    in each table corresponds to the same loan across all four.
    """
    app_dates = pd.to_datetime(
        dataset.loan_applications["application_date"]
    ).reset_index(drop=True)
    fin_as_of = pd.to_datetime(dataset.financial_profiles["as_of_date"]).reset_index(
        drop=True
    )
    credit_as_of = pd.to_datetime(dataset.credit_history["as_of_date"]).reset_index(
        drop=True
    )

    assert (fin_as_of <= app_dates).all()
    assert (credit_as_of <= app_dates).all()


def test_correlation_signs_match_label_design(dataset) -> None:
    """Income should correlate negatively, utilization/DTI positively, with default."""
    default = dataset.loan_outcomes["default_12m"].to_numpy()
    utilization = dataset.credit_history["credit_utilization"].to_numpy()
    dti = _dti(dataset.financial_profiles)
    income = _loan_level_income(dataset)

    assert np.corrcoef(income, default)[0, 1] < 0
    assert np.corrcoef(utilization, default)[0, 1] > 0
    assert np.corrcoef(dti, default)[0, 1] > 0


def test_no_single_feature_achieves_high_auc(dataset) -> None:
    """No single raw feature should trivially determine the label (sanity check)."""
    default = dataset.loan_outcomes["default_12m"].to_numpy()
    features = {
        "credit_utilization": dataset.credit_history["credit_utilization"].to_numpy(),
        "dti": _dti(dataset.financial_profiles),
        "annual_income": _loan_level_income(dataset),
        "previous_defaults": dataset.credit_history["previous_defaults"].to_numpy(),
        "loan_amount": dataset.loan_applications["loan_amount"].to_numpy(),
        "credit_history_months": dataset.credit_history[
            "credit_history_months"
        ].to_numpy(),
    }
    for name, feature in features.items():
        auc = _auc(default, feature)
        directional_auc = max(auc, 1 - auc)
        assert (
            directional_auc < 0.75
        ), f"{name} alone achieves AUC={directional_auc:.3f}"


def test_injected_error_counts_match_manifest(config: dict) -> None:
    """Manifest counts must match the configured injection rates and the actual data."""
    n = 4000
    ds = generate_dataset(config, seed=7, n_customers=n)
    inj_cfg = config["data_quality_injection"]
    manifest = ds.injection_manifest

    assert manifest["out_of_range_age"]["count"] == round(
        n * inj_cfg["out_of_range_age_rate"]
    )
    assert manifest["negative_financial_value"]["count"] == round(
        n * inj_cfg["negative_financial_value_rate"]
    )
    assert manifest["inconsistent_income"]["count"] == round(
        n * inj_cfg["inconsistent_income_rate"]
    )
    assert manifest["duplicate_customer"]["count"] == round(
        n * inj_cfg["duplicate_customer_rate"]
    )

    n_financial = len(ds.financial_profiles)
    assert manifest["missing_value"]["count"] == round(
        n_financial * inj_cfg["missing_value_rate"]
    )
    n_credit = len(ds.credit_history)
    assert manifest["impossible_utilization"]["count"] == round(
        n_credit * inj_cfg["impossible_utilization_rate"]
    )

    bad_ages = {5, 12, 17, 105, 130, 150}
    assert (
        int(ds.customers["age"].isin(bad_ages).sum())
        == manifest["out_of_range_age"]["count"]
    )
    assert (
        int(ds.financial_profiles["total_assets"].isna().sum())
        == manifest["missing_value"]["count"]
    )
    assert (
        int((~ds.credit_history["credit_utilization"].between(0, 2)).sum())
        == manifest["impossible_utilization"]["count"]
    )
    assert (
        ds.customers["customer_id"].duplicated().sum()
        == manifest["duplicate_customer"]["count"]
    )
