import numpy as np
import pytest

from bankguard.evaluate import evaluate, weighted_sample


def make(n, rate, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    scores = rng.random(n) + 0.6 * y
    return y, scores


def test_sampling_every_negative_reproduces_the_full_metrics():
    y, s = make(5_000, 0.02, seed=0)
    idx, w = weighted_sample(y, n_negatives=int((y == 0).sum()))
    full, weighted = evaluate(y, s, "m"), evaluate(y[idx], s[idx], "m", sample_weight=w)
    for key in ["pr_auc", "recall_at_0.1%_fpr", "recall_at_1.0%_fpr"]:
        assert weighted[key] == pytest.approx(full[key])


def test_weighted_sample_estimates_the_full_metrics():
    y, s = make(200_000, 0.005, seed=1)
    idx, w = weighted_sample(y, n_negatives=40_000)
    assert len(idx) == y.sum() + 40_000
    assert w.sum() == pytest.approx(len(y))
    full, weighted = evaluate(y, s, "m"), evaluate(y[idx], s[idx], "m", sample_weight=w)
    assert weighted["pr_auc"] == pytest.approx(full["pr_auc"], abs=0.02)
    assert weighted["recall_at_1.0%_fpr"] == pytest.approx(full["recall_at_1.0%_fpr"], abs=0.03)