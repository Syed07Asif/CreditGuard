"""CLI entrypoint that idempotently applies db/schema.sql to the configured database.

Usage: python -m creditguard.db.init_db
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from creditguard.config import get_settings
from creditguard.db.engine import get_engine

SCHEMA_SQL_PATH = Path(__file__).resolve().parents[3] / "db" / "schema.sql"


def apply_schema() -> None:
    """Create the configured schema (if needed) and apply db/schema.sql idempotently."""
    settings = get_settings()
    engine = get_engine()
    schema_sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{settings.db_schema}"'))
        conn.exec_driver_sql(schema_sql)


def main() -> None:
    """Apply the database schema and report success."""
    apply_schema()
    print(f"Applied schema from {SCHEMA_SQL_PATH}")


if __name__ == "__main__":
    main()
