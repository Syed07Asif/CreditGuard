"""Page 1's applicant intake form: grouped expandable sections, widgets
carrying the same min/max the API schema enforces (`PredictionRequest`,
`creditguard.api.schemas`), and the three "Load sample" demo buttons.

Sample-loading writes into `st.session_state` and reruns the script (the
only way to change a widget's displayed value in Streamlit) rather than
mutating widgets after they're created.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from creditguard.dashboard.state import FORM_STATE_KEY, SAMPLE_PAYLOADS

GENDERS = ["MALE", "FEMALE", "OTHER"]
MARITAL_STATUSES = ["SINGLE", "MARRIED", "DIVORCED", "WIDOWED"]
EDUCATIONS = ["HIGH_SCHOOL", "GRADUATE", "POSTGRADUATE", "DOCTORATE"]
EMPLOYMENT_TYPES = ["SALARIED", "SELF_EMPLOYED", "BUSINESS_OWNER", "UNEMPLOYED"]
LOAN_TYPES = ["PERSONAL", "HOME", "AUTO", "EDUCATION", "BUSINESS", "CREDIT_CARD"]


def _current() -> dict[str, Any]:
    return st.session_state.get(FORM_STATE_KEY, SAMPLE_PAYLOADS["Low risk"])


def render_sample_loader_buttons() -> None:
    """Three buttons outside the form (a `st.form` only allows one submit
    button) that load a demo payload into session state and rerun.
    """
    st.caption("Load a sample applicant to try the form:")
    cols = st.columns(3)
    for col, name in zip(cols, SAMPLE_PAYLOADS, strict=True):
        if col.button(f"Load sample: {name.lower()}", width="stretch"):
            st.session_state[FORM_STATE_KEY] = dict(SAMPLE_PAYLOADS[name])
            st.rerun()


def render_applicant_form() -> dict[str, Any] | None:
    """Renders the grouped applicant form. Returns the submitted payload
    (matching `PredictionRequest`'s field names/types) on submit, else
    `None`.
    """
    current = _current()

    with st.form("applicant_scoring_form"):
        with st.expander("Identifiers (optional)", expanded=False):
            customer_id = st.text_input(
                "Customer ID", value=current.get("customer_id", "") or ""
            )
            loan_id = st.text_input("Loan ID", value=current.get("loan_id", "") or "")

        with st.expander("Personal", expanded=True):
            c1, c2, c3 = st.columns(3)
            age = c1.number_input(
                "Age", min_value=18, max_value=100, value=int(current["age"])
            )
            dependents = c2.number_input(
                "Dependents",
                min_value=0,
                max_value=20,
                value=int(current["dependents"]),
            )
            city_tier = c3.number_input(
                "City tier", min_value=1, max_value=3, value=int(current["city_tier"])
            )
            c4, c5, c6 = st.columns(3)
            gender = c4.selectbox(
                "Gender", GENDERS, index=GENDERS.index(current["gender"])
            )
            marital_status = c5.selectbox(
                "Marital status",
                MARITAL_STATUSES,
                index=MARITAL_STATUSES.index(current["marital_status"]),
            )
            education = c6.selectbox(
                "Education", EDUCATIONS, index=EDUCATIONS.index(current["education"])
            )

        with st.expander("Employment & Income", expanded=True):
            c1, c2 = st.columns(2)
            employment_type = c1.selectbox(
                "Employment type",
                EMPLOYMENT_TYPES,
                index=EMPLOYMENT_TYPES.index(current["employment_type"]),
            )
            employment_years = c2.number_input(
                "Employment years",
                min_value=0.0,
                max_value=60.0,
                value=float(current["employment_years"]),
                step=0.5,
            )
            c3, c4 = st.columns(2)
            monthly_income = c3.number_input(
                "Monthly income",
                min_value=1.0,
                value=float(current["monthly_income"]),
                step=1000.0,
            )
            annual_income = c4.number_input(
                "Annual income",
                min_value=1.0,
                value=float(current["annual_income"]),
                step=10000.0,
                help="Must be within 5% of 12 x monthly income.",
            )

        with st.expander("Financial Position", expanded=True):
            c1, c2, c3 = st.columns(3)
            monthly_expenses = c1.number_input(
                "Monthly expenses",
                min_value=0.0,
                value=float(current["monthly_expenses"]),
                step=1000.0,
            )
            existing_loan_count = c2.number_input(
                "Existing loan count",
                min_value=0,
                value=int(current["existing_loan_count"]),
            )
            existing_loan_amount = c3.number_input(
                "Existing loan amount",
                min_value=0.0,
                value=float(current["existing_loan_amount"]),
                step=10000.0,
            )
            c4, c5, c6 = st.columns(3)
            monthly_emi = c4.number_input(
                "Monthly EMI",
                min_value=0.0,
                value=float(current["monthly_emi"]),
                step=1000.0,
            )
            savings_balance = c5.number_input(
                "Savings balance",
                min_value=0.0,
                value=float(current["savings_balance"]),
                step=10000.0,
            )
            total_assets = c6.number_input(
                "Total assets",
                min_value=0.0,
                value=float(current["total_assets"]),
                step=10000.0,
            )
            total_liabilities = st.number_input(
                "Total liabilities",
                min_value=0.0,
                value=float(current["total_liabilities"]),
                step=10000.0,
            )

        with st.expander("Credit History", expanded=True):
            c1, c2, c3 = st.columns(3)
            credit_history_months = c1.number_input(
                "Credit history (months)",
                min_value=0,
                value=int(current["credit_history_months"]),
            )
            num_credit_accounts = c2.number_input(
                "Number of credit accounts",
                min_value=0,
                value=int(current["num_credit_accounts"]),
            )
            credit_utilization = c3.slider(
                "Credit utilization",
                min_value=0.0,
                max_value=2.0,
                value=float(current["credit_utilization"]),
                step=0.01,
            )
            c4, c5 = st.columns(2)
            total_credit_limit = c4.number_input(
                "Total credit limit",
                min_value=0.0,
                value=float(current["total_credit_limit"]),
                step=10000.0,
            )
            total_outstanding = c5.number_input(
                "Total outstanding",
                min_value=0.0,
                value=float(current["total_outstanding"]),
                step=10000.0,
                help="Must be at most 1.5x total credit limit.",
            )
            c6, c7, c8, c9 = st.columns(4)
            previous_defaults = c6.number_input(
                "Previous defaults",
                min_value=0,
                value=int(current["previous_defaults"]),
            )
            late_payments_12m = c7.number_input(
                "Late payments (12m)",
                min_value=0,
                value=int(current["late_payments_12m"]),
            )
            missed_payments_12m = c8.number_input(
                "Missed payments (12m)",
                min_value=0,
                value=int(current["missed_payments_12m"]),
            )
            active_loans = c9.number_input(
                "Active loans", min_value=0, value=int(current["active_loans"])
            )
            closed_loans = st.number_input(
                "Closed loans", min_value=0, value=int(current["closed_loans"])
            )

        with st.expander("Loan Request", expanded=True):
            c1, c2 = st.columns(2)
            loan_type = c1.selectbox(
                "Loan type", LOAN_TYPES, index=LOAN_TYPES.index(current["loan_type"])
            )
            loan_purpose = c2.text_input("Loan purpose", value=current["loan_purpose"])
            c3, c4, c5 = st.columns(3)
            loan_amount = c3.number_input(
                "Loan amount",
                min_value=1.0,
                value=float(current["loan_amount"]),
                step=10000.0,
            )
            loan_tenure_months = c4.number_input(
                "Loan tenure (months)",
                min_value=1,
                value=int(current["loan_tenure_months"]),
            )
            interest_rate = c5.number_input(
                "Interest rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(current["interest_rate"]),
                step=0.1,
            )

        submitted = st.form_submit_button(
            "Score application", width="stretch", type="primary"
        )

    if not submitted:
        return None

    return {
        "customer_id": customer_id or None,
        "loan_id": loan_id or None,
        "age": int(age),
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "employment_type": employment_type,
        "dependents": int(dependents),
        "employment_years": float(employment_years),
        "annual_income": float(annual_income),
        "monthly_income": float(monthly_income),
        "city_tier": int(city_tier),
        "monthly_expenses": float(monthly_expenses),
        "existing_loan_count": int(existing_loan_count),
        "existing_loan_amount": float(existing_loan_amount),
        "monthly_emi": float(monthly_emi),
        "savings_balance": float(savings_balance),
        "total_assets": float(total_assets),
        "total_liabilities": float(total_liabilities),
        "credit_history_months": int(credit_history_months),
        "num_credit_accounts": int(num_credit_accounts),
        "total_credit_limit": float(total_credit_limit),
        "total_outstanding": float(total_outstanding),
        "credit_utilization": float(credit_utilization),
        "previous_defaults": int(previous_defaults),
        "late_payments_12m": int(late_payments_12m),
        "missed_payments_12m": int(missed_payments_12m),
        "active_loans": int(active_loans),
        "closed_loans": int(closed_loans),
        "loan_type": loan_type,
        "loan_amount": float(loan_amount),
        "loan_tenure_months": int(loan_tenure_months),
        "interest_rate": float(interest_rate),
        "loan_purpose": loan_purpose,
    }
