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