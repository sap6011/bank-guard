"""The served model reproduces its evaluated scores on every demo transfer.

Needs the trained model and the demo bundle, which are not in git, so these
tests are skipped in CI and run on the machine that built them.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from bankguard.serving import DemoScorer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "sequence"
BUNDLE = ROOT / "models" / "demo" / "bundle.npz"

pytestmark = pytest.mark.skipif(
    not (BUNDLE.exists() and (MODEL_DIR / "preprocessing.json").exists()),
    reason="needs the trained model and the demo bundle",
)


@pytest.fixture(scope="module")
def scorer():
    return DemoScorer(MODEL_DIR, BUNDLE)


@pytest.fixture(scope="module")
def scores(scorer):
    return scorer.score(np.arange(len(scorer)))


def test_scores_match_evaluation(scorer, scores):
    assert np.abs(scores - scorer.data["evaluated_score"]).max() < 1e-5


def test_percentile_follows_score(scorer, scores):
    pct = scorer.percentile(scores)
    assert pct.min() >= 0 and pct.max() <= 100
    assert np.all(np.diff(pct[np.argsort(scores)]) >= 0)


def test_stricter_alert_flags_a_subset(scorer, scores):
    alerts = scorer.alerts(scores)
    assert alerts.shape == (len(scores), len(scorer.data["alert_fpr"]))
    assert (alerts[:, 1] <= alerts[:, 0]).all()

@pytest.fixture(scope="module")
def busiest(scorer):
    """The demo transfer with the most past transfers on both sides."""
    d = scorer.data
    return int(np.argmax(d["sender_mask"].sum(1) + d["receiver_mask"].sum(1)))


def test_unchanged_what_if_matches_score(scorer, scores, busiest):
    i = busiest
    amount = float(scorer.data["detail_amount"][scorer.data["tx"][i]])
    assert abs(scorer.what_if(i) - scores[i]) < 1e-5
    assert abs(scorer.what_if(i, amount=amount) - scores[i]) < 1e-5


def test_explanation_matches_removal(scorer, scores, busiest):
    i = busiest
    base, table = scorer.explain(i)
    d = scorer.data
    assert len(table) == d["sender_mask"][i].sum() + d["receiver_mask"][i].sum()
    assert abs(base - scores[i]) < 1e-5
    top = table.iloc[0]
    without = scorer.what_if(i, removed=[(top.side, top.slot)])
    assert abs(base - top.effect - without) < 1e-5