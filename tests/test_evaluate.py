import numpy as np
import pytest

from bankguard.evaluate import evaluate, recall_at_fpr

# 2 laundering rows and 10 normal rows. One normal row outranks the second laundering row.
Y = np.array([1, 1] + [0] * 10)
SCORES = np.array([0.9, 0.7, 0.8] + [0.1] * 9)


def test_recall_depends_on_the_alert_budget():
    # No false alarms allowed: only the top laundering row is caught.
    assert recall_at_fpr(Y, SCORES, 0.0) == 0.5
    # Allow 1 of the 10 normal rows to be flagged: both are caught.
    assert recall_at_fpr(Y, SCORES, 0.1) == 1.0


def test_evaluate_returns_one_results_row():
    row = evaluate(Y, SCORES, "toy")
    assert list(row) == ["model", "pr_auc", "recall_at_0.1%_fpr", "recall_at_1.0%_fpr"]
    # Average precision: (1/1 + 2/3) / 2
    assert row["pr_auc"] == pytest.approx(5 / 6)


def test_perfect_ranking_scores_one():
    row = evaluate(Y, Y.astype(float), "perfect")
    assert row["pr_auc"] == pytest.approx(1.0)
    assert row["recall_at_0.1%_fpr"] == 1.0


def test_a_model_that_knows_nothing_scores_the_base_rate():
    row = evaluate(Y, np.zeros(len(Y)), "constant")
    assert row["pr_auc"] == pytest.approx(Y.mean())
    assert row["recall_at_1.0%_fpr"] == 0.0