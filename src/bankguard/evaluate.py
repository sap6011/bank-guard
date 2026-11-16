"""Metrics for a rare-event detector."""

from sklearn.metrics import average_precision_score, roc_curve


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