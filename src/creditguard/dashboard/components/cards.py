"""Small reusable Streamlit renderers: friendly API-error cards, risk/
recommendation badges, and a KPI-row helper. Everything here renders
directly via `st.*` rather than returning markup strings, except the two
badge helpers (used inline inside a larger `st.markdown` block on Page 1).
"""

from __future__ import annotations

import streamlit as st

from creditguard.dashboard.api_client import (
    ApiAuthError,
    ApiClientError,
    ApiConnectionError,
    ApiNotFoundError,
    ApiRateLimitError,
    ApiServerError,
    ApiValidationError,
)
from creditguard.dashboard.state import RECOMMENDATION_COLORS, RISK_BAND_COLORS


def render_api_error(exc: ApiClientError, *, context: str = "") -> None:
    """Render one friendly card for `exc` -- never let a raw traceback (or
    the API's own internal error text, which the API itself never leaks
    either) reach the screen.
    """
    prefix = f"{context}: " if context else ""
    if isinstance(exc, ApiConnectionError):
        st.error(
            f"{prefix}Can't reach the CreditGuard API. Is it running? " f"(`{exc}`)"
        )
    elif isinstance(exc, ApiAuthError):
        st.error(
            f"{prefix}The dashboard's API key was rejected. Check the "
            "`API_KEY` environment variable matches the API's."
        )
    elif isinstance(exc, ApiValidationError):
        st.warning(f"{prefix}The request was rejected: {exc}")
        for item in exc.detail:
            loc = ".".join(str(part) for part in item.get("loc", []))
            st.caption(f"- {loc}: {item.get('msg')}")
    elif isinstance(exc, ApiNotFoundError):
        st.warning(f"{prefix}Not found: {exc}")
    elif isinstance(exc, ApiRateLimitError):
        st.warning(
            f"{prefix}Rate limit exceeded -- please wait a moment and try again."
        )
    elif isinstance(exc, ApiServerError):
        st.error(f"{prefix}The CreditGuard API returned an error: {exc}")
    else:
        st.error(f"{prefix}Unexpected error: {exc}")


def _badge_html(label: str, color: str) -> str:
    return (
        f'<span style="background-color:{color};color:white;padding:4px 12px;'
        f'border-radius:12px;font-weight:600;font-size:0.85rem;">{label}</span>'
    )


def risk_badge_html(risk_category: str) -> str:
    color = RISK_BAND_COLORS.get(risk_category, "#666666")
    return _badge_html(risk_category.replace("_", " "), color)


def recommendation_badge_html(recommendation: str) -> str:
    color = RECOMMENDATION_COLORS.get(recommendation, "#666666")
    return _badge_html(recommendation, color)


def render_kpi_row(kpis: list[tuple[str, str]]) -> None:
    """`kpis` is a list of `(label, value)` pairs, laid out as equal-width
    `st.metric` columns.
    """
    if not kpis:
        return
    columns = st.columns(len(kpis))
    for col, (label, value) in zip(columns, kpis, strict=True):
        col.metric(label, value)


def render_model_caption(model_id: str, model_version: str) -> None:
    st.caption(f"Model: `{model_id}` (v{model_version})")


def render_sidebar_status() -> None:
    """Cross-cutting sidebar (every page calls this): API connection
    status (green/red) and the active model version -- per the Phase 9
    brief's "consistent... sidebar" requirement.
    """
    from creditguard.dashboard.api_client import get_client

    with st.sidebar:
        st.markdown("### API status")
        client = get_client()
        try:
            client.health()
        except ApiClientError as exc:
            st.markdown("🔴 **Unreachable**")
            st.caption(str(exc))
            return

        st.markdown("🟢 **Connected**")
        try:
            info = client.model_info()
            st.caption(f"Active model: `{info['model_id']}`")
            st.caption(f"Version: {info['model_version']}")
        except ApiClientError:
            st.caption("Model info unavailable")
