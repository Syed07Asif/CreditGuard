"""Page 2: Portfolio Analytics (FR-018, FR-019, FR-020).

Aggregates `GET /api/v1/predictions` (paginated client-side via
`fetch_all_predictions`) into KPIs, distribution/trend charts and segment
breakdowns. "Default rate" here always means the *predicted* rate -- the
share of applications at/above the active model's chosen threshold -- not
an observed 12-month outcome, which the API has no endpoint for (a live
scoring predictions log has no ground truth yet). Segment breakdowns by
loan type / age / income / employment use the Phase 9 columns added to
`predictions` (see docs/api.md) -- `None` for any prediction logged before
that migration.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from creditguard.dashboard.api_client import (
    ApiClientError,
    cached_model_info,
    fetch_all_predictions,
)
from creditguard.dashboard.components import cards, charts, tables
from creditguard.dashboard.state import (
    AGE_BAND_ORDER,
    INCOME_BAND_ORDER,
    RISK_BAND_ORDER,
    age_band,
    income_band,
)

st.title("Portfolio Analytics")
st.caption("Aggregate view over predictions logged by the CreditGuard API.")

try:
    chosen_threshold = float(cached_model_info()["chosen_threshold"])
except ApiClientError as exc:
    cards.render_api_error(exc, context="Could not load the active model")
    st.stop()

# -- sidebar filters ----------------------------------------------------------

st.sidebar.markdown("### Filters")
default_start = date.today() - timedelta(days=90)
date_range = st.sidebar.date_input("Date range", value=(default_start, date.today()))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, date.today()

loan_type_filter = st.sidebar.multiselect(
    "Loan type", ["PERSONAL", "HOME", "AUTO", "EDUCATION", "BUSINESS", "CREDIT_CARD"]
)
risk_filter = st.sidebar.multiselect("Risk category", list(RISK_BAND_ORDER))
recommendation_filter = st.sidebar.multiselect(
    "Recommendation", ["APPROVE", "REVIEW", "REJECT"]
)

try:
    rows = fetch_all_predictions(
        date_from=f"{start_date}T00:00:00", date_to=f"{end_date}T23:59:59"
    )
except ApiClientError as exc:
    cards.render_api_error(exc, context="Could not load predictions")
    st.stop()

frame = pd.DataFrame(rows)
if frame.empty:
    st.info("No predictions logged yet for the selected date range.")
    st.stop()

if loan_type_filter:
    frame = frame[frame["loan_type"].isin(loan_type_filter)]
if risk_filter:
    frame = frame[frame["risk_category"].isin(risk_filter)]
if recommendation_filter:
    frame = frame[frame["recommendation"].isin(recommendation_filter)]

if frame.empty:
    st.info("No predictions match the current filters.")
    st.stop()

frame["created_at"] = pd.to_datetime(frame["created_at"])
frame["predicted_default"] = frame["default_probability"] >= chosen_threshold

# -- KPI row --------------------------------------------------------------

total = len(frame)
approve_n = int((frame["recommendation"] == "APPROVE").sum())
review_n = int((frame["recommendation"] == "REVIEW").sum())
reject_n = int((frame["recommendation"] == "REJECT").sum())
high_risk_n = int(frame["risk_category"].isin(["HIGH", "VERY_HIGH"]).sum())
predicted_default_rate = float(frame["predicted_default"].mean())

cards.render_kpi_row(
    [
        ("Total applications", f"{total:,}"),
        ("Approved", f"{approve_n:,} ({approve_n / total:.1%})"),
        ("Review", f"{review_n:,} ({review_n / total:.1%})"),
        ("Rejected", f"{reject_n:,} ({reject_n / total:.1%})"),
    ]
)
cards.render_kpi_row(
    [
        ("Avg. credit score", f"{frame['credit_score'].mean():.0f}"),
        (
            "Avg. default probability",
            tables.format_percentage(frame["default_probability"].mean()),
        ),
        ("High-risk share", f"{high_risk_n / total:.1%}"),
        ("Predicted default rate", tables.format_percentage(predicted_default_rate)),
    ]
)

st.divider()

# -- distribution charts ---------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    st.pyplot(charts.score_distribution_histogram(frame["credit_score"]))
with col2:
    st.pyplot(
        charts.risk_category_bar(frame["risk_category"].value_counts(), RISK_BAND_ORDER)
    )

daily = (
    frame.assign(date=frame["created_at"].dt.date)
    .groupby("date")
    .agg(n=("prediction_id", "count"), default_rate=("predicted_default", "mean"))
    .reset_index()
)
st.pyplot(charts.applications_over_time(daily))

# -- segment breakdowns ------------------------------------------------------

st.subheader("Default rate by segment")
seg_col1, seg_col2 = st.columns(2)
with seg_col1:
    by_loan_type = frame.dropna(subset=["loan_type"])
    if not by_loan_type.empty:
        st.pyplot(
            charts.rate_by_segment_bar(
                by_loan_type.groupby("loan_type")["predicted_default"].mean(),
                "Predicted default rate",
                "By loan type",
            )
        )
    by_employment = frame.dropna(subset=["employment_type"])
    if not by_employment.empty:
        st.pyplot(
            charts.rate_by_segment_bar(
                by_employment.groupby("employment_type")["predicted_default"].mean(),
                "Predicted default rate",
                "By employment type",
            )
        )
with seg_col2:
    by_income = frame.dropna(subset=["annual_income"]).assign(
        income_band=lambda d: d["annual_income"].apply(income_band)
    )
    if not by_income.empty:
        st.pyplot(
            charts.rate_by_segment_bar(
                by_income.groupby("income_band")["predicted_default"].mean(),
                "Predicted default rate",
                "By income band",
                order=INCOME_BAND_ORDER,
            )
        )
    by_age = frame.dropna(subset=["age"]).assign(
        age_band=lambda d: d["age"].apply(age_band)
    )
    if not by_age.empty:
        st.pyplot(
            charts.rate_by_segment_bar(
                by_age.groupby("age_band")["predicted_default"].mean(),
                "Predicted default rate",
                "By age band",
                order=AGE_BAND_ORDER,
            )
        )

if frame["loan_type"].notna().any():
    st.subheader("Approval rate by loan type")
    approval_by_type = (
        frame.dropna(subset=["loan_type"])
        .assign(is_approve=lambda d: d["recommendation"] == "APPROVE")
        .groupby("loan_type")["is_approve"]
        .mean()
    )
    st.pyplot(
        charts.rate_by_segment_bar(
            approval_by_type, "Approval rate", "Approval rate by loan type"
        )
    )

# -- segment analysis table --------------------------------------------------

st.subheader("Segment analysis")
segment_dim = st.selectbox(
    "Segment by", ["loan_type", "risk_category", "recommendation", "employment_type"]
)
segment_table = (
    frame.dropna(subset=[segment_dim])
    .groupby(segment_dim)
    .agg(
        n=("prediction_id", "count"),
        avg_score=("credit_score", "mean"),
        avg_default_probability=("default_probability", "mean"),
        predicted_default_rate=("predicted_default", "mean"),
        approval_rate=("recommendation", lambda s: (s == "APPROVE").mean()),
    )
    .round(4)
    .reset_index()
    .sort_values("n", ascending=False)
)
tables.render_segment_table(segment_table, filename=f"segment_by_{segment_dim}.csv")

# -- searchable predictions log ---------------------------------------------

st.subheader("Recent predictions")
display_columns = [
    "created_at",
    "loan_id",
    "customer_id",
    "loan_type",
    "credit_score",
    "risk_category",
    "recommendation",
    "default_probability",
]
tables.render_predictions_table(
    frame[display_columns].sort_values("created_at", ascending=False),
    search_columns=["loan_id", "customer_id"],
)

with st.expander("Look up full application record"):
    loan_ids = frame["loan_id"].dropna().unique().tolist()
    selected_loan_id = st.selectbox("Loan ID", loan_ids) if loan_ids else None
    if selected_loan_id and st.button("Fetch full record"):
        try:
            from creditguard.dashboard.api_client import cached_get_application

            application = cached_get_application(selected_loan_id)
        except ApiClientError as exc:
            cards.render_api_error(exc, context="Could not load application")
        else:
            st.json(application)
