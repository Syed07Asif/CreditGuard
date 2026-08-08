"""SQLAlchemy engine and session factory for CreditGuard's PostgreSQL database."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from creditguard.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine built from the application settings."""
    settings = get_settings()
    return create_engine(
        settings.sqlalchemy_dsn,
        pool_pre_ping=True,
        future=True,
        connect_args={"options": f"-csearch_path={settings.db_schema}"},
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Return a cached sessionmaker bound to the application engine."""
    return sessionmaker(
        bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
    )


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a transactional session that commits on success and rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
