"""Table rendering + small formatting helpers shared across pages.

Formatting helpers are plain, side-effect-free functions (testable without
a Streamlit runtime); the render_* functions call `st.*` directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from creditguard.dashboard.state import RISK_BAND_COLORS


def format_currency(value: float | int | None, symbol: str = "₹") -> str:
    """`1234567.8` -> `'₹1,234,568'`. `None`/`NaN` -> `'--'`."""
    if value is None or pd.isna(value):
        return "--"
    return f"{symbol}{value:,.0f}"


def format_percentage(value: float | None, decimals: int = 1) -> str:
    """A `[0, 1]` fraction -> a percentage string, e.g. `0.0583` -> `'5.8%'`."""
    if value is None or pd.isna(value):
        return "--"
    return f"{value * 100:.{decimals}f}%"


def format_ratio(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.{decimals}f}"


def score_band_color(risk_category: str | None) -> str:
    return RISK_BAND_COLORS.get(risk_category or "", "#666666")


def render_csv_download(
    frame: pd.DataFrame, filename: str, label: str = "Download CSV"
) -> None:
    if frame.empty:
        return
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def render_segment_table(frame: pd.DataFrame, filename: str) -> None:
    """A sortable segment-breakdown table (Streamlit's own dataframe grid
    already supports column-header sorting) plus a CSV export button.
    """
    if frame.empty:
        st.info("No data for the current filters.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)
    render_csv_download(frame, filename)


def render_predictions_table(frame: pd.DataFrame, search_columns: list[str]) -> None:
    """A searchable predictions log: a text filter applied (case-
    insensitive substring match) across `search_columns`, then an
    `st.dataframe` grid the user can sort/scroll.
    """
    if frame.empty:
        st.info("No predictions logged yet for the current filters.")
        return

    query = st.text_input(
        "Search (loan ID / customer ID)",
        key="predictions_search",
        placeholder="e.g. LOAN-...",
    )
    filtered = frame
    if query:
        mask = pd.Series(False, index=frame.index)
        for column in search_columns:
            if column in frame.columns:
                mask = mask | frame[column].astype(str).str.contains(
                    query, case=False, na=False
                )
        filtered = frame[mask]

    st.dataframe(filtered, width="stretch", hide_index=True)
    st.caption(f"{len(filtered)} of {len(frame)} predictions shown.")
