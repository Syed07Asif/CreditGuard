"""Tests for creditguard.config: settings loading and validation."""

from __future__ import annotations

import os

import pytest
from pydantic_settings import SettingsConfigDict

from creditguard.config import ConfigurationError, Settings, get_settings


def test_settings_loads_from_environment() -> None:
    """Settings should read all configured values from the process environment."""
    settings = Settings()  # type: ignore[call-arg]
    assert settings.db_host == os.environ["DB_HOST"]
    assert settings.db_name == os.environ["DB_NAME"]
    assert settings.seed == 42
    assert settings.env == "test"


def test_settings_computes_sqlalchemy_dsn() -> None:
    """The computed DSN should embed host, port and database name."""
    settings = Settings()  # type: ignore[call-arg]
    dsn = settings.sqlalchemy_dsn
    assert dsn.startswith("postgresql+psycopg://")
    assert settings.db_host in dsn
    assert settings.db_name in dsn


def test_get_settings_is_cached() -> None:
    """get_settings() should return the same cached instance on repeated calls."""
    assert get_settings() is get_settings()


def test_missing_required_variable_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing required variable should raise a clear ConfigurationError.

    Disables .env file loading for this test so a real local .env (which
    developers are expected to have) can't mask the missing variable.
    """
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    monkeypatch.setattr(
        Settings, "model_config", SettingsConfigDict(env_file=None, extra="ignore")
    )
    get_settings.cache_clear()
    try:
        with pytest.raises(ConfigurationError, match="DB_PASSWORD"):
            get_settings()
    finally:
        get_settings.cache_clear()
