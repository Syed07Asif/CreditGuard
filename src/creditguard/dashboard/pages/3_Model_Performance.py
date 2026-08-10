"""Page 3: Model Performance.

Scalar metrics come from `GET /api/v1/model/info`; curves/tables come from
`GET /api/v1/model/performance` (Phase 9 addition -- see docs/api.md and
`creditguard.models.performance`'s docstring for why that data is
backfilled once offline rather than recomputed per page view). If the
backfill hasn't been run for the active model, that endpoint 503s and this
page shows a friendly explanation instead of blank charts.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from creditguard.dashboard.api_client import (
    ApiClientError,
    cached_model_info,
    cached_model_performance,
    cached_model_versions,
)
from creditguard.dashboard.components import cards, charts

st.title("Model Performance")
st.caption(
    "Evaluation metrics for the active model, measured on its held-out test split."
)

try:
    info = cached_model_info()
except ApiClientError as exc:
    cards.render_api_error(exc, context="Could not load model info")
    st.stop()

# -- model card ---------------------------------------------------------

st.subheader("Active model")
card_cols = st.columns(4)
card_cols[0].markdown(f"**Model ID**\n\n`{info['model_id']}`")
card_cols[1].markdown(f"**Version**\n\n{info['model_version']}")
card_cols[2].markdown(f"**Algorithm**\n\n{info['algorithm']}")
card_cols[3].markdown(f"**Training date**\n\n{str(info['training_date'])[:10]}")
card_cols2 = st.columns(4)
card_cols2[0].markdown(f"**Dataset version**\n\n`{info['dataset_version']}`")
card_cols2[1].markdown(f"**Feature count**\n\n{info['feature_count']}")
card_cols2[2].markdown(f"**Chosen threshold**\n\n{info['chosen_threshold']:.4f}")
card_cols2[3].markdown(f"**Active**\n\n{'Yes' if info['is_active'] else 'No'}")

st.divider()

# -- metric tiles ---------------------------------------------------------

metrics = info["metrics"]
st.subheader("Metrics (test split)")
cards.render_kpi_row(
    [
        ("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.4f}"),
        ("PR-AUC", f"{metrics.get('pr_auc', float('nan')):.4f}"),
        ("KS statistic", f"{metrics.get('ks_statistic', float('nan')):.4f}"),
        ("Brier score", f"{metrics.get('brier_score', float('nan')):.4f}"),
    ]
)
cards.render_kpi_row(
    [
        ("Precision", f"{metrics.get('precision', float('nan')):.4f}"),
        ("Recall", f"{metrics.get('recall', float('nan')):.4f}"),
        ("F1", f"{metrics.get('f1', float('nan')):.4f}"),
        ("Calibration slope", f"{metrics.get('calibration_slope', float('nan')):.4f}"),
    ]
)

st.divider()

# -- curves/tables (Phase 9 backfilled data) --------------------------------

try:
    performance = cached_model_performance()
except ApiClientError as exc:
    cards.render_api_error(
        exc,
        context="Performance curves unavailable -- run "
        "`python -m creditguard.models.performance` to backfill them "
        "for the active model",
    )
else:
    st.subheader("Curves")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(
            charts.roc_curve_chart(
                performance["roc_curve"]["fpr"],
                performance["roc_curve"]["tpr"],
                metrics.get("roc_auc"),
            )
        )
    with col2:
        st.pyplot(
            charts.pr_curve_chart(
                performance["pr_curve"]["precision"],
                performance["pr_curve"]["recall"],
                metrics.get("pr_auc"),
            )
        )
    with col3:
        st.pyplot(charts.confusion_matrix_heatmap(performance["confusion_matrix"]))

    col4, col5 = st.columns(2)
    with col4:
        st.pyplot(
            charts.calibration_chart(
                performance["calibration_curve"]["mean_predicted"],
                performance["calibration_curve"]["fraction_positive"],
            )
        )
    with col5:
        st.pyplot(charts.lift_gains_chart(performance["lift_gains"]))

    st.subheader("Global feature importance")
    st.pyplot(charts.feature_importance_chart(performance["feature_importance"]))

st.divider()

# -- version comparison table ------------------------------------------------

st.subheader("Version comparison")
try:
    versions = cached_model_versions()["versions"]
except ApiClientError as exc:
    cards.render_api_error(exc, context="Could not load model versions")
else:
    rows = []
    for version in versions:
        row = {
            "model_id": version["model_id"],
            "model_version": version["model_version"],
            "algorithm": version["algorithm"],
            "training_date": str(version["training_date"])[:10],
            "is_active": version["is_active"],
        }
        for key in ("roc_auc", "pr_auc", "ks_statistic", "brier_score", "f1"):
            row[key] = version["metrics"].get(key)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
