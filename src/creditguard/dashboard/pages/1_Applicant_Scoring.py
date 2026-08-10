"""Page 1: Applicant Scoring (FR-017).

Submits one application to `POST /api/v1/predict` (score + explanation
factors) and `POST /api/v1/explain` (full per-feature breakdown, used only
for the ratio-vs-portfolio-median panel), then renders the result. No
business logic lives here -- every number displayed comes straight from
the API response.
"""

from __future__ import annotations

import json

import streamlit as st

from creditguard.dashboard.api_client import ApiClientError, get_client
from creditguard.dashboard.components import cards, forms, tables
from creditguard.dashboard.components.charts import score_gauge, shap_contribution_chart
from creditguard.dashboard.state import RESULT_STATE_KEY

RATIO_FIELDS: list[tuple[str, str]] = [
    ("dti", "DTI"),
    ("post_loan_dti", "Post-loan DTI"),
    ("credit_utilization", "Utilisation"),
    ("loan_to_income", "LTI"),
]

st.title("Applicant Scoring")
st.caption("Score a single loan application through the live CreditGuard API.")

forms.render_sample_loader_buttons()
payload = forms.render_applicant_form()

if payload is not None:
    client = get_client()
    with st.spinner("Scoring application..."):
        try:
            prediction = client.predict(payload)
        except ApiClientError as exc:
            cards.render_api_error(exc, context="Scoring failed")
            prediction = None
        else:
            try:
                explanation = client.explain(payload)
            except ApiClientError:
                explanation = None
            st.session_state[RESULT_STATE_KEY] = {
                "prediction": prediction,
                "explanation": explanation,
            }

result_state = st.session_state.get(RESULT_STATE_KEY)

if result_state is not None:
    prediction = result_state["prediction"]
    explanation = result_state.get("explanation")

    st.divider()
    st.header("Result")

    col_gauge, col_summary = st.columns([1, 2])
    with col_gauge:
        st.pyplot(score_gauge(prediction["credit_score"], prediction["risk_category"]))

    with col_summary:
        probability = prediction["default_probability"]
        st.markdown(
            f"### Default probability: {tables.format_percentage(probability)} "
            f"<span style='font-size:0.9rem;color:#666;'>"
            f"(raw: {probability:.4f})</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{cards.risk_badge_html(prediction['risk_category'])} &nbsp; "
            f"{cards.recommendation_badge_html(prediction['recommendation'])}",
            unsafe_allow_html=True,
        )
        st.markdown("**Triggered rules:**")
        if prediction["triggered_rules"]:
            for rule in prediction["triggered_rules"]:
                st.write(f"- {rule}")
        else:
            st.write("_None_")

    st.subheader("Feature contributions")
    st.pyplot(
        shap_contribution_chart(
            prediction["top_risk_factors"], prediction["top_positive_factors"]
        )
    )

    st.subheader("Reason codes")
    for factor in [
        *prediction["top_risk_factors"],
        *prediction["top_positive_factors"],
    ]:
        st.write(f"- {factor['description']}")

    if explanation is not None:
        st.subheader("Key ratios vs. portfolio median")
        contributions_by_feature = {
            item["feature"]: item for item in explanation["feature_contributions"]
        }
        ratio_cols = st.columns(len(RATIO_FIELDS))
        for col, (key, label) in zip(ratio_cols, RATIO_FIELDS, strict=True):
            factor = contributions_by_feature.get(key)
            if factor is None:
                col.metric(label, "--")
                continue
            median = factor.get("benchmark_median")
            col.metric(
                label,
                tables.format_ratio(factor["value"]),
                delta=(
                    f"median {tables.format_ratio(median)}"
                    if median is not None
                    else None
                ),
                delta_color="off",
            )

    cards.render_model_caption(prediction["model_id"], prediction["model_version"])

    st.download_button(
        "Download result as JSON",
        data=json.dumps(prediction, indent=2),
        file_name=f"prediction_{prediction['request_id']}.json",
        mime="application/json",
    )
