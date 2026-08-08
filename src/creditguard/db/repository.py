"""Thin, typed repository classes providing data access over the CreditGuard schema."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import pandas as pd
from sqlalchemy import insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from creditguard.db.engine import get_engine, get_session
from creditguard.db.models import (
    Base,
    CreditHistory,
    Customer,
    DataQualityIssue,
    DriftReport,
    FinancialProfile,
    LoanApplication,
    LoanOutcome,
    ModelRegistry,
    MonitoringMetric,
    Prediction,
)

ModelT = TypeVar("ModelT", bound=Base)


def _row_to_dict(row: Base) -> dict[str, Any]:
    """Convert an ORM row instance into a plain dict of column values."""
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


class BaseRepository(Generic[ModelT]):
    """Generic insert/read/upsert repository for a single ORM model."""

    model: type[ModelT]
    pk_column: str

    def insert_many(self, records: list[dict[str, Any]]) -> int:
        """Bulk-insert records and return the number of rows inserted."""
        if not records:
            return 0
        with get_session() as session:
            session.execute(insert(self.model), records)
        return len(records)

    def get_by_id(self, id_value: Any) -> dict[str, Any] | None:
        """Fetch a single row by primary key, returned as a dict, or None if absent."""
        with get_session() as session:
            pk = getattr(self.model, self.pk_column)
            row = session.execute(
                select(self.model).where(pk == id_value)
            ).scalar_one_or_none()
            return _row_to_dict(row) if row is not None else None

    def fetch_dataframe(
        self,
        sql: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Fetch rows as a DataFrame, either via raw SQL or equality filters."""
        if sql is not None:
            with get_engine().connect() as conn:
                return pd.read_sql(text(sql), conn)
        with get_session() as session:
            stmt = select(self.model)
            if filters:
                for column_name, value in filters.items():
                    stmt = stmt.where(getattr(self.model, column_name) == value)
            rows = session.execute(stmt).scalars().all()
            return pd.DataFrame([_row_to_dict(row) for row in rows])

    def upsert(self, records: list[dict[str, Any]]) -> int:
        """Insert records, updating existing rows on primary-key conflict."""
        if not records:
            return 0
        stmt = pg_insert(self.model).values(records)
        update_columns = {
            column.name: getattr(stmt.excluded, column.name)
            for column in self.model.__table__.columns
            if column.name != self.pk_column
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.pk_column], set_=update_columns
        )
        with get_session() as session:
            session.execute(stmt)
        return len(records)


class CustomerRepository(BaseRepository[Customer]):
    """Repository for the customers table."""

    model = Customer
    pk_column = "customer_id"


class LoanApplicationRepository(BaseRepository[LoanApplication]):
    """Repository for the loan_applications table."""

    model = LoanApplication
    pk_column = "loan_id"


class FinancialProfileRepository(BaseRepository[FinancialProfile]):
    """Repository for the financial_profiles table."""

    model = FinancialProfile
    pk_column = "profile_id"


class CreditHistoryRepository(BaseRepository[CreditHistory]):
    """Repository for the credit_history table."""

    model = CreditHistory
    pk_column = "credit_id"


class LoanOutcomeRepository(BaseRepository[LoanOutcome]):
    """Repository for the loan_outcomes label table."""

    model = LoanOutcome
    pk_column = "loan_id"


class ModelRegistryRepository(BaseRepository[ModelRegistry]):
    """Repository for the model_registry table."""

    model = ModelRegistry
    pk_column = "model_id"


class PredictionRepository(BaseRepository[Prediction]):
    """Repository for the predictions table."""

    model = Prediction
    pk_column = "prediction_id"


class DataQualityIssueRepository(BaseRepository[DataQualityIssue]):
    """Repository for the data_quality_issues table."""

    model = DataQualityIssue
    pk_column = "issue_id"


class MonitoringMetricRepository(BaseRepository[MonitoringMetric]):
    """Repository for the monitoring_metrics table."""

    model = MonitoringMetric
    pk_column = "metric_id"


class DriftReportRepository(BaseRepository[DriftReport]):
    """Repository for the drift_reports table."""

    model = DriftReport
    pk_column = "report_id"
