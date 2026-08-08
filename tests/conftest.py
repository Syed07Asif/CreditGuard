"""Shared pytest fixtures: test environment defaults and database setup/teardown."""

from __future__ import annotations

import os

# Env vars must be set before creditguard modules are imported anywhere, since
# Settings validation happens on first use.
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "creditguard_test")
os.environ.setdefault("DB_USER", "creditguard")
os.environ.setdefault("DB_PASSWORD", "creditguard")
os.environ.setdefault("DB_SCHEMA", "public")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("SEED", "42")

from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from creditguard.db.engine import get_engine  # noqa: E402
from creditguard.db.init_db import apply_schema  # noqa: E402
from creditguard.db.models import Base  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database() -> None:
    """Apply the schema once per test session before any test runs."""
    apply_schema()


@pytest.fixture(autouse=True)
def _clean_tables() -> Iterator[None]:
    """Truncate all application tables before each test so tests stay isolated."""
    engine = get_engine()
    table_names = [table.name for table in Base.metadata.sorted_tables]
    with engine.begin() as conn:
        quoted = ", ".join(f'"{name}"' for name in table_names)
        conn.execute(text(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE"))
    yield
