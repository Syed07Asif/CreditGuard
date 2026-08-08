"""Tests for creditguard.data.ingest: row counts and quarantine behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select

from creditguard.data.generator import generate_dataset, load_config
from creditguard.data.ingest import ingest_dataset
from creditguard.data.versioning import GeneratedDataset, write_dataset
from creditguard.db.engine import get_session
from creditguard.db.models import (
    CreditHistory,
    Customer,
    DataQualityIssue,
    FinancialProfile,
    LoanApplication,
    LoanOutcome,
)

N_CUSTOMERS = 2000
DATASET_VERSION = "ds_test_ingest"


@pytest.fixture(scope="module")
def generated(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[GeneratedDataset, Path]:
    """Generate a small dataset once and write it to a temporary raw data directory."""
    config = load_config("config/data_generation.yaml")
    dataset = generate_dataset(config, seed=11, n_customers=N_CUSTOMERS)
    data_root = tmp_path_factory.mktemp("data") / "raw"
    write_dataset(dataset, DATASET_VERSION, data_root)
    return dataset, data_root


def _count(model: type) -> int:
    with get_session() as session:
        return session.execute(select(func.count()).select_from(model)).scalar_one()


def test_ingest_loads_expected_row_counts_and_quarantines_bad_rows(
    generated: tuple[GeneratedDataset, Path],
) -> None:
    """Ingest must load every clean row and quarantine exactly the injected bad ones."""
    dataset, data_root = generated
    summary = ingest_dataset(
        DATASET_VERSION, truncate=True, data_dir=data_root, progress=False
    )

    manifest = dataset.injection_manifest
    bad_customer_ids = set(manifest["out_of_range_age"]["customer_ids"]) | set(
        manifest["negative_financial_value"]["customer_ids"]
    )

    expected_customers_quarantined = (
        manifest["out_of_range_age"]["count"]
        + manifest["negative_financial_value"]["count"]
        + manifest["duplicate_customer"]["count"]
    )
    assert summary.tables["customers"].attempted == len(dataset.customers)
    assert summary.tables["customers"].quarantined == expected_customers_quarantined
    assert summary.tables["customers"].inserted == (
        len(dataset.customers) - expected_customers_quarantined
    )

    # Rows belonging to a dropped (out-of-range-age / negative-income) customer
    # cascade-fail via FK violation even though they aren't themselves corrupted.
    cascade_loans = int(
        dataset.loan_applications["customer_id"].isin(bad_customer_ids).sum()
    )
    assert summary.tables["loan_applications"].quarantined == cascade_loans
    assert summary.tables["loan_outcomes"].quarantined == cascade_loans

    fin_bad_mask = dataset.financial_profiles[
        "total_assets"
    ].isna() | dataset.financial_profiles["customer_id"].isin(bad_customer_ids)
    assert summary.tables["financial_profiles"].quarantined == int(fin_bad_mask.sum())

    credit_bad_mask = ~dataset.credit_history["credit_utilization"].between(
        0, 2
    ) | dataset.credit_history["customer_id"].isin(bad_customer_ids)
    assert summary.tables["credit_history"].quarantined == int(credit_bad_mask.sum())

    assert _count(Customer) == summary.tables["customers"].inserted
    assert _count(LoanApplication) == summary.tables["loan_applications"].inserted
    assert _count(FinancialProfile) == summary.tables["financial_profiles"].inserted
    assert _count(CreditHistory) == summary.tables["credit_history"].inserted
    assert _count(LoanOutcome) == summary.tables["loan_outcomes"].inserted

    total_quarantined = sum(result.quarantined for result in summary.tables.values())
    assert _count(DataQualityIssue) == total_quarantined


def test_truncate_reingest_is_idempotent(
    generated: tuple[GeneratedDataset, Path],
) -> None:
    """Re-ingesting the same dataset with --truncate must reproduce identical counts."""
    _, data_root = generated
    first = ingest_dataset(
        DATASET_VERSION, truncate=True, data_dir=data_root, progress=False
    )
    second = ingest_dataset(
        DATASET_VERSION, truncate=True, data_dir=data_root, progress=False
    )

    for table_name in first.tables:
        assert first.tables[table_name] == second.tables[table_name]
    assert _count(DataQualityIssue) == sum(
        result.quarantined for result in second.tables.values()
    )
