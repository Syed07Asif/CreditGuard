"""Shared constants and `st.session_state` helpers used across dashboard
pages: the colour palette (matching `creditguard.eda.plots`' Okabe-Ito
palette per the Phase 9 brief's "consistent theme" requirement), risk-band/
recommendation colours, the three demo sample payloads for Page 1, and
simple display-only age/income bucketing for Page 2's segment charts.

The age/income bands here are fixed, human-readable cutoffs for *display*
grouping only -- not the model's own learned quantile `age_band`/
`income_band` features (`creditguard.features.behavioural`), which the
dashboard has no access to (and doesn't need to reproduce) since it never
touches the feature pipeline directly.
"""

from __future__ import annotations

from typing import Any

# Okabe-Ito palette, same as creditguard.eda.plots.OKABE_ITO -- kept as an
# independent literal (not imported) so the dashboard never depends on
# creditguard.eda, which pulls in matplotlib/seaborn plotting internals
# unrelated to a Streamlit page.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # grey
)
COLOR_POSITIVE = OKABE_ITO[2]  # bluish green
COLOR_NEGATIVE = OKABE_ITO[1]  # vermillion
COLOR_NEUTRAL = OKABE_ITO[7]  # grey
COLOR_VOLUME = OKABE_ITO[0]  # blue

RISK_BAND_ORDER: tuple[str, ...] = ("VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH")
RISK_BAND_COLORS: dict[str, str] = {
    "VERY_LOW": "#1a7a4c",
    "LOW": "#5aa15a",
    "MODERATE": "#e6a817",
    "HIGH": "#e0722a",
    "VERY_HIGH": "#c1352c",
}
# score -> band, mirroring config/scoring.yaml's risk_categories.bands
# (VERY_LOW 750-900 / LOW 700-749 / MODERATE 650-699 / HIGH 550-649 /
# VERY_HIGH 300-549) -- duplicated as a display constant rather than fetched
# from the API because the gauge needs to colour the full 300-900 axis, not
# just the applicant's own band.
SCORE_BAND_RANGES: tuple[tuple[str, int, int], ...] = (
    ("VERY_HIGH", 300, 549),
    ("HIGH", 550, 649),
    ("MODERATE", 650, 699),
    ("LOW", 700, 749),
    ("VERY_LOW", 750, 900),
)

RECOMMENDATION_COLORS: dict[str, str] = {
    "APPROVE": "#1a7a4c",
    "REVIEW": "#e6a817",
    "REJECT": "#c1352c",
}


def age_band(age: int) -> str:
    """Fixed display bucket for Page 2's "by age band" segment views."""
    if age < 26:
        return "18-25"
    if age < 36:
        return "26-35"
    if age < 46:
        return "36-45"
    if age < 56:
        return "46-55"
    if age < 66:
        return "56-65"
    return "66+"


AGE_BAND_ORDER: tuple[str, ...] = ("18-25", "26-35", "36-45", "46-55", "56-65", "66+")


def income_band(annual_income: float) -> str:
    """Fixed display bucket (INR) for Page 2's "by income band" segment
    views. Cut points chosen to spread this project's synthetic population
    (see `reports/eda/findings.md`) roughly evenly, not a real income scale.
    """
    if annual_income < 300_000:
        return "<3L"
    if annual_income < 600_000:
        return "3-6L"
    if annual_income < 1_000_000:
        return "6-10L"
    if annual_income < 2_000_000:
        return "10-20L"
    return "20L+"


INCOME_BAND_ORDER: tuple[str, ...] = ("<3L", "3-6L", "6-10L", "10-20L", "20L+")


# -- Page 1 demo sample payloads ---------------------------------------------
# Every payload satisfies PredictionRequest's cross-field checks (annual
# income ~= 12x monthly income, employment_years <= age-16, total_outstanding
# <= 1.5x total_credit_limit, monthly_expenses+monthly_emi <= 2x
# monthly_income) so "Load sample" always submits cleanly.

SAMPLE_LOW_RISK: dict[str, Any] = {
    "age": 35,
    "gender": "MALE",
    "marital_status": "MARRIED",
    "education": "POSTGRADUATE",
    "employment_type": "SALARIED",
    "dependents": 1,
    "employment_years": 8.0,
    "annual_income": 960000.0,
    "monthly_income": 80000.0,
    "city_tier": 1,
    "monthly_expenses": 20000.0,
    "existing_loan_count": 0,
    "existing_loan_amount": 0.0,
    "monthly_emi": 0.0,
    "savings_balance": 500000.0,
    "total_assets": 2000000.0,
    "total_liabilities": 0.0,
    "credit_history_months": 96,
    "num_credit_accounts": 4,
    "total_credit_limit": 500000.0,
    "total_outstanding": 50000.0,
    "credit_utilization": 0.10,
    "previous_defaults": 0,
    "late_payments_12m": 0,
    "missed_payments_12m": 0,
    "active_loans": 0,
    "closed_loans": 3,
    "loan_type": "PERSONAL",
    "loan_amount": 200000.0,
    "loan_tenure_months": 36,
    "interest_rate": 10.5,
    "loan_purpose": "HOME_IMPROVEMENT",
}

SAMPLE_MEDIUM_RISK: dict[str, Any] = {
    "age": 29,
    "gender": "FEMALE",
    "marital_status": "MARRIED",
    "education": "GRADUATE",
    "employment_type": "SALARIED",
    "dependents": 2,
    "employment_years": 4.0,
    "annual_income": 420000.0,
    "monthly_income": 35000.0,
    "city_tier": 2,
    "monthly_expenses": 18000.0,
    "existing_loan_count": 1,
    "existing_loan_amount": 150000.0,
    "monthly_emi": 6000.0,
    "savings_balance": 40000.0,
    "total_assets": 300000.0,
    "total_liabilities": 150000.0,
    "credit_history_months": 30,
    "num_credit_accounts": 2,
    "total_credit_limit": 150000.0,
    "total_outstanding": 90000.0,
    "credit_utilization": 0.60,
    "previous_defaults": 0,
    "late_payments_12m": 2,
    "missed_payments_12m": 1,
    "active_loans": 1,
    "closed_loans": 1,
    "loan_type": "AUTO",
    "loan_amount": 300000.0,
    "loan_tenure_months": 48,
    "interest_rate": 13.5,
    "loan_purpose": "VEHICLE_PURCHASE",
}

SAMPLE_HIGH_RISK: dict[str, Any] = {
    "age": 24,
    "gender": "MALE",
    "marital_status": "SINGLE",
    "education": "HIGH_SCHOOL",
    "employment_type": "SELF_EMPLOYED",
    "dependents": 0,
    "employment_years": 1.0,
    "annual_income": 240000.0,
    "monthly_income": 20000.0,
    "city_tier": 3,
    "monthly_expenses": 15000.0,
    "existing_loan_count": 3,
    "existing_loan_amount": 250000.0,
    "monthly_emi": 9000.0,
    "savings_balance": 2000.0,
    "total_assets": 50000.0,
    "total_liabilities": 250000.0,
    "credit_history_months": 8,
    "num_credit_accounts": 2,
    "total_credit_limit": 100000.0,
    "total_outstanding": 92000.0,
    "credit_utilization": 0.92,
    "previous_defaults": 2,
    "late_payments_12m": 5,
    "missed_payments_12m": 3,
    "active_loans": 3,
    "closed_loans": 0,
    "loan_type": "PERSONAL",
    "loan_amount": 150000.0,
    "loan_tenure_months": 24,
    "interest_rate": 18.0,
    "loan_purpose": "DEBT_CONSOLIDATION",
}

SAMPLE_PAYLOADS: dict[str, dict[str, Any]] = {
    "Low risk": SAMPLE_LOW_RISK,
    "Medium risk": SAMPLE_MEDIUM_RISK,
    "High risk": SAMPLE_HIGH_RISK,
}

FORM_STATE_KEY = "applicant_form_values"
RESULT_STATE_KEY = "last_scoring_result"
