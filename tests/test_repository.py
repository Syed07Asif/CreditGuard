"""Tests for creditguard.db.repository: insert/read round-trips via the repository."""

from __future__ import annotations

from datetime import date

from creditguard.db.repository import CustomerRepository, LoanApplicationRepository


def _customer_record(customer_id: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = dict(
        customer_id=customer_id,
        age=30,
        gender="F",
        marital_status="SINGLE",
        dependents=0,
        education="GRADUATE",
        employment_type="SALARIED",
        employment_years=5,
        annual_income=500000,
        monthly_income=40000,
        city_tier=2,
    )
    record.update(overrides)
    return record


def test_customer_insert_and_get_by_id_round_trip() -> None:
    """A customer inserted via the repository should read back with the same values."""
    repo = CustomerRepository()
    repo.insert_many([_customer_record("CUST-100", age=42, annual_income=1200000)])

    fetched = repo.get_by_id("CUST-100")

    assert fetched is not None
    assert fetched["customer_id"] == "CUST-100"
    assert fetched["age"] == 42
    assert float(fetched["annual_income"]) == 1200000


def test_fetch_dataframe_filters_by_column() -> None:
    """fetch_dataframe should return only rows matching the given equality filters."""
    repo = CustomerRepository()
    repo.insert_many(
        [
            _customer_record("CUST-200", city_tier=3),
            _customer_record("CUST-201", city_tier=1),
        ]
    )

    df = repo.fetch_dataframe(filters={"city_tier": 3})

    assert len(df) == 1
    assert df.iloc[0]["customer_id"] == "CUST-200"


def test_fetch_dataframe_with_raw_sql() -> None:
    """fetch_dataframe should also support executing raw SQL as a DataFrame."""
    repo = CustomerRepository()
    repo.insert_many([_customer_record("CUST-210")])

    df = repo.fetch_dataframe(sql="SELECT customer_id FROM customers")

    assert "CUST-210" in df["customer_id"].tolist()


def test_upsert_updates_existing_row() -> None:
    """Upserting a record with an existing primary key should update it in place."""
    repo = CustomerRepository()
    repo.insert_many([_customer_record("CUST-300", age=25)])

    repo.upsert([_customer_record("CUST-300", age=26)])

    fetched = repo.get_by_id("CUST-300")
    assert fetched is not None
    assert fetched["age"] == 26


def test_loan_application_round_trip_with_foreign_key() -> None:
    """A loan application referencing an existing customer should round-trip."""
    customer_repo = CustomerRepository()
    loan_repo = LoanApplicationRepository()

    customer_repo.insert_many([_customer_record("CUST-400")])
    loan_repo.insert_many(
        [
            {
                "loan_id": "LOAN-400",
                "customer_id": "CUST-400",
                "loan_type": "PERSONAL",
                "loan_amount": 200000,
                "loan_tenure_months": 24,
                "interest_rate": 12.5,
                "loan_purpose": "MEDICAL",
                "application_date": date(2026, 1, 15),
                "status": "PENDING",
            }
        ]
    )

    fetched = loan_repo.get_by_id("LOAN-400")

    assert fetched is not None
    assert fetched["customer_id"] == "CUST-400"
    assert fetched["status"] == "PENDING"
