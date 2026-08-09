"""Threshold optimisation: 0.5 is not a decision, it's a default -- this
module picks the decision threshold deliberately, from a configurable cost
matrix where a false negative (approving a defaulter) costs far more than a
false positive (rejecting a good applicant), and reports the max-F1 and
max-Youden's-J alternatives alongside it for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from creditguard.models.evaluate import confusion_counts_at_threshold


@dataclass(frozen=True)
class CostMatrix:
    """Cost of each confusion-matrix outcome. `cost_false_negative` (an
    approved defaulter) should dominate `cost_false_positive` (a rejected
    good applicant) for credit risk -- see `config/model_config.yaml`.
    """

    cost_false_negative: float
    cost_false_positive: float
    cost_true_positive: float = 0.0
    cost_true_negative: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CostMatrix:
        """Build from the `cost_matrix` section of `config/model_config.yaml`."""
        return cls(
            cost_false_negative=float(config["cost_false_negative"]),
            cost_false_positive=float(config["cost_false_positive"]),
            cost_true_positive=float(config.get("cost_true_positive", 0.0)),
            cost_true_negative=float(config.get("cost_true_negative", 0.0)),
        )


def expected_cost_at_threshold(
    y_true: Any, y_prob: Any, threshold: float, cost_matrix: CostMatrix
) -> float:
    """Total cost of every prediction at `threshold`, summed over the
    confusion matrix counts.
    """
    counts = confusion_counts_at_threshold(y_true, y_prob, threshold)
    return (
        counts["fn"] * cost_matrix.cost_false_negative
        + counts["fp"] * cost_matrix.cost_false_positive
        + counts["tp"] * cost_matrix.cost_true_positive
        + counts["tn"] * cost_matrix.cost_true_negative
    )


def find_min_cost_threshold(
    y_true: Any, y_prob: Any, cost_matrix: CostMatrix, n_thresholds: int = 1000
) -> tuple[float, float]:
    """The threshold in `[0, 1]` minimising `expected_cost_at_threshold`."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    costs = [
        expected_cost_at_threshold(y_true, y_prob, t, cost_matrix) for t in thresholds
    ]
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])


def find_max_f1_threshold(y_true: Any, y_prob: Any) -> tuple[float, float]:
    """The threshold maximising F1, read off the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall + 1e-12),
        0.0,
    )
    # precision_recall_curve returns one more point than thresholds (the
    # last point is precision=1, recall=0 with no corresponding threshold).
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    if len(thresholds) == 0:
        return 0.5, float(f1[0]) if len(f1) else 0.0
    return float(thresholds[best_idx]), float(f1[best_idx])


def find_max_youden_j_threshold(y_true: Any, y_prob: Any) -> tuple[float, float]:
    """The threshold maximising Youden's J (`tpr - fpr`), read off the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx]), float(j[best_idx])


def optimise_threshold(
    y_true: Any, y_prob: Any, cost_matrix: CostMatrix, n_thresholds: int = 1000
) -> dict[str, Any]:
    """All three candidate thresholds, plus the chosen one (minimum expected
    cost, since that's the operating point Phase 7's lending decision needs).
    """
    min_cost_t, min_cost = find_min_cost_threshold(
        y_true, y_prob, cost_matrix, n_thresholds
    )
    max_f1_t, max_f1 = find_max_f1_threshold(y_true, y_prob)
    youden_t, youden_j = find_max_youden_j_threshold(y_true, y_prob)
    return {
        "min_cost": {"threshold": min_cost_t, "expected_cost": min_cost},
        "max_f1": {"threshold": max_f1_t, "f1": max_f1},
        "max_youden_j": {"threshold": youden_t, "youden_j": youden_j},
        "chosen_threshold": min_cost_t,
        "chosen_rationale": (
            "minimises expected cost under the configured cost matrix "
            f"(false negative cost={cost_matrix.cost_false_negative:g} vs. "
            f"false positive cost={cost_matrix.cost_false_positive:g}) -- "
            "approving a defaulter is treated as materially more expensive "
            "than rejecting a good applicant, so 0.5 / max-F1 / max-Youden's-J "
            "are reported for comparison but not used as the operating point."
        ),
    }
