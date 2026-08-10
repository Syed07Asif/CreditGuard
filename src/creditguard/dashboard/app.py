"""Phase 9 dashboard entry point / router.

    streamlit run src/creditguard/dashboard/app.py

Uses `st.navigation`/`st.Page` (Streamlit's current page-router API) rather
than the older implicit `pages/`-directory auto-discovery -- the installed
Streamlit version no longer wires that up on its own (confirmed directly:
`streamlit.source_util.get_pages`, the function the old mechanism relied
on, no longer exists in this version). The four page files under `pages/`
keep the brief-mandated names/paths; this router just points at them, and
also owns the one-time `st.set_page_config` call and the shared sidebar
status check every page used to duplicate.
"""

from __future__ import annotations

import streamlit as st

from creditguard.dashboard.components.cards import render_sidebar_status

st.set_page_config(page_title="CreditGuard", page_icon="\U0001f4b3", layout="wide")


def _home() -> None:
    st.title("CreditGuard: Credit Risk Scoring & Monitoring")
    st.caption(
        "Educational simulation on synthetic data -- not a real lending system. "
        "See docs/scoring_methodology.md for what the score/category/"
        "recommendation mean."
    )
    st.markdown("""
Use the pages in the left sidebar:

- **Applicant Scoring** -- score one application and see its credit score, risk
  category, lending recommendation and a SHAP-based explanation.
- **Portfolio Analytics** -- explore scored applications in aggregate: KPIs, score
  distribution, risk/recommendation mix, and trends over time.
- **Model Performance** -- the active model's evaluation metrics, curves and
  global feature importance.
- **Monitoring** -- drift and data-quality monitoring (Phase 10; this page is a
  placeholder layout until that phase lands).

Every page here is a pure HTTP client of the Phase 8 API
(`creditguard.dashboard.api_client`) -- nothing in this dashboard imports the
model, the feature pipeline or the scoring engine directly.
""")


PAGES = [
    st.Page(_home, title="Home", icon="\U0001f3e0", default=True),
    st.Page(
        "pages/1_Applicant_Scoring.py",
        title="Applicant Scoring",
        icon="\U0001f4dd",
    ),
    st.Page(
        "pages/2_Portfolio_Analytics.py",
        title="Portfolio Analytics",
        icon="\U0001f4ca",
    ),
    st.Page(
        "pages/3_Model_Performance.py",
        title="Model Performance",
        icon="\U0001f3af",
    ),
    st.Page("pages/4_Monitoring.py", title="Monitoring", icon="\U0001f6f0"),
]

page = st.navigation(PAGES)
render_sidebar_status()
page.run()
