import numpy as np
import pandas as pd

from bankguard.evaluate import evaluate
from bankguard.model import fit_categories, fit_gbm, gbm_frame, predict

COLUMNS = ["amount", "noise", "format"]


def make_data(n, seed):
    """Laundering is more likely for large ACH payments; `noise` carries no signal."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "amount": rng.normal(size=n),
        "noise": rng.normal(size=n),
        "format": rng.choice(["ACH", "Cheque", "Wire"], size=n),
    })
    risk = 0.02 + 0.3 * ((X["amount"] > 1) & (X["format"] == "ACH"))
    y = (rng.random(n) < risk).to_numpy().astype(int)
    return X, y


def test_categories_come_from_train_in_sorted_order():
    train = pd.DataFrame({"format": ["Wire", "ACH", None, "ACH"]})
    assert fit_categories(train, ["format"]) == {"format": ["ACH", "Wire"]}


def test_frame_keeps_codes_fixed_and_unseen_values_become_missing():
    categories = {"format": ["ACH", "Cheque", "Wire"]}
    later = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "noise": [0.0, 0.0, 0.0],
                          "format": ["Wire", "Cheque", "Bitcoin"]})
    frame = gbm_frame(later, COLUMNS, categories)
    assert list(frame.columns) == COLUMNS
    # Cheque stays code 1 even though ACH is absent from this batch. Unseen is -1 (missing).
    assert frame["format"].cat.codes.tolist() == [2, 1, -1]


def test_model_learns_the_planted_signal_and_is_reproducible():
    X_train, y_train = make_data(6000, seed=0)
    X_val, y_val = make_data(3000, seed=1)
    categories = fit_categories(X_train, ["format"])

    model = fit_gbm(X_train, y_train, X_val, y_val, COLUMNS, categories)
    scores = predict(model, X_val, COLUMNS, categories)

    assert scores.min() >= 0 and scores.max() <= 1
    # Far above the base rate, which is what a model that learned nothing would score.
    assert evaluate(y_val, scores, "toy")["pr_auc"] > 3 * y_val.mean()

    again = fit_gbm(X_train, y_train, X_val, y_val, COLUMNS, categories)
    assert np.array_equal(scores, predict(again, X_val, COLUMNS, categories))