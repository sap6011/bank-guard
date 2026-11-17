"""Metrics for a rare-event detector."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

from bankguard.data import NO_PATTERN


def recall_at_fpr(y_true, scores, max_fpr: float) -> float:
    """Share of laundering caught when at most `max_fpr` of normal rows are flagged."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= max_fpr].max())


def evaluate(y_true, scores, name: str, fprs=(0.001, 0.01)) -> dict:
    """PR-AUC plus recall at each false positive rate, as one results row."""
    result = {"model": name, "pr_auc": float(average_precision_score(y_true, scores))}
    for f in fprs:
        result[f"recall_at_{f:.1%}_fpr"] = recall_at_fpr(y_true, scores, f)
    return result

HUB_THRESHOLD = 1000  # sender payments in the previous 3 days
GROUPS = ["patterned", "unpatterned, quiet sender", "unpatterned, hub sender"]


def laundering_group(typology, sender_out_3d, hub_threshold: int = HUB_THRESHOLD) -> np.ndarray:
    """Split rows into the three groups used in the error analysis.

    Only laundering rows are meaningful; normal rows get a group too but are ignored.
    """
    patterned = np.asarray(typology) != NO_PATTERN
    hub = np.asarray(sender_out_3d) >= hub_threshold
    return np.where(
        patterned, GROUPS[0], np.where(hub, GROUPS[2], GROUPS[1])
    )


def recall_by_group(y_true, scores, group, fprs=(0.01, 0.001)) -> pd.DataFrame:
    """Recall within each laundering group, at alert budgets set on normal rows."""
    y_true, scores, group = np.asarray(y_true), np.asarray(scores), np.asarray(group)
    masks = {g: (y_true == 1) & (group == g) for g in GROUPS}
    table = pd.DataFrame(index=GROUPS)
    table["laundering"] = [int(m.sum()) for m in masks.values()]
    for fpr in fprs:
        # The score above which the top `fpr` share of normal rows sit.
        threshold = np.quantile(scores[y_true == 0], 1 - fpr)
        caught = scores > threshold
        table[f"recall_at_{fpr:.1%}_fpr"] = [
            float(caught[m].mean()) if m.any() else np.nan for m in masks.values()
        ]
    return table