"""Page 4: Monitoring -- layout wired now, data filled in by Phase 10.

`monitoring_metrics`/`drift_reports` tables already exist in the schema
(Phase 1), but no monitoring *computation* and no API endpoint exposing
them exist yet (that's Phase 10's job, per CLAUDE.md's build-phase
checklist -- this phase is explicitly out of scope for computing anything
here). Every section below is an intentional "no monitoring runs yet"
empty state, not a broken/errored one.
"""

from __future__ import annotations

import streamlit as st

st.title("Monitoring")
st.caption(
    "Drift and data-quality monitoring. Layout wired in Phase 9; monitoring "
    "computation and its API endpoint are Phase 10 work."
)
st.info(
    "No monitoring runs yet -- this page will populate once Phase 10 adds "
    "scheduled drift/quality checks and a `GET /api/v1/monitoring/*` endpoint "
    "for the dashboard to read from.",
    icon="ℹ️",
)

st.divider()

st.subheader("Feature drift status")
st.caption(
    "Per-feature drift status (OK / WARNING / DRIFT) and PSI values, most recent run."
)
st.info("No monitoring runs yet.")

st.subheader("Prediction distribution drift")
st.caption(
    "Scored-population score/probability distribution vs. the training baseline."
)
st.info("No monitoring runs yet.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model performance over time")
    st.caption("Rolling ROC-AUC/PR-AUC/calibration on newly-labelled outcomes.")
    st.info("No monitoring runs yet.")
with col2:
    st.subheader("Data quality trend")
    st.caption(
        "Validation rule violation counts over time (Phase 3's rule engine, scheduled)."
    )
    st.info("No monitoring runs yet.")
