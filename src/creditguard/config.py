"""Typed application configuration loaded from environment variables and .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid at startup."""


class Settings(BaseSettings):
    """Application settings sourced from environment variables (or a local .env)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_host: str = Field(alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(alias="DB_NAME")
    db_user: str = Field(alias="DB_USER")
    db_password: str = Field(alias="DB_PASSWORD")
    db_schema: str = Field(default="public", alias="DB_SCHEMA")

    mlflow_tracking_uri: str = Field(alias="MLFLOW_TRACKING_URI")
    api_key: str = Field(alias="API_KEY")
    env: Literal["dev", "test", "prod"] = Field(default="dev", alias="ENV")
    seed: int = Field(default=42, alias="SEED")

    model_dir: Path = Field(default=Path("models/artifacts"), alias="MODEL_DIR")
    reports_dir: Path = Field(default=Path("reports"), alias="REPORTS_DIR")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")

    # Phase 8 (API)
    rate_limit_rpm: int = Field(default=100, alias="RATE_LIMIT_RPM")
    cors_origins: str = Field(default="http://localhost:8501", alias="CORS_ORIGINS")

    # Phase 10 (Monitoring) -- operational/deployment tunables. Domain
    # thresholds (PSI bands, degradation tolerance, retraining triggers, ...)
    # live in config/monitoring.yaml instead, per this module's own
    # env-vars-are-for-secrets-and-deployment-context convention.
    monitoring_interval_minutes: int = Field(
        default=60, alias="MONITORING_INTERVAL_MINUTES"
    )
    alert_webhook_url: str | None = Field(default=None, alias="ALERT_WEBHOOK_URL")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        """`cors_origins` (comma-separated) split into a list for
        `CORSMiddleware` -- the Streamlit dashboard's origin(s), configured
        rather than hard-coded so `.env` alone controls it per environment.
        """
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_dsn(self) -> str:
        """Build the SQLAlchemy connection string for the configured database."""
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached, validated Settings instance.

    Raises:
        ConfigurationError: if a required environment variable is missing or invalid.
    """
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        missing = [
            str(error["loc"][0]) for error in exc.errors() if error["type"] == "missing"
        ]
        if missing:
            raise ConfigurationError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            ) from exc
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc
