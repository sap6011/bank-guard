import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from bankguard.features import CATEGORICAL, FEATURES
from bankguard.sequence import build_events
from bankguard.sequence_model import (
    SequenceModel,
    encode_rows,
    load_models,
    score_rows,
)

TYPES = ["Corporation", "Individual"]


def make_features(n, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.poisson(3, n).astype(float) for c in FEATURES})
    for c in CATEGORICAL:
        X[c] = rng.choice(TYPES, n)
    return X


def make_prep(n_seeds=2):
    columns = [c for c in FEATURES if c not in CATEGORICAL]
    columns += [f"{c}_{t}" for c in CATEGORICAL for t in TYPES]
    return {
        "history_length": 8,
        "seeds": list(range(n_seeds)),
        "categories": {c: TYPES for c in CATEGORICAL},
        "row_columns": columns,
        "row_mean": [0.0] * len(columns),
        "row_std": [1.0] * len(columns),
        "amount_mean": [0.0, 0.0],
        "amount_std": [1.0, 1.0],
    }


def test_encode_rows_shape_and_unseen_category():
    prep = make_prep()
    X = make_features(5)
    X.loc[0, "sender_entity_type"] = "Unknown"
    out = encode_rows(X, prep)
    assert out.shape == (5, len(prep["row_columns"]))
    onehot = [prep["row_columns"].index(f"sender_entity_type_{t}") for t in TYPES]
    assert not out[0, onehot].any()  # a type never seen in training gets all zeros
    assert out[1:, onehot].sum(axis=1).tolist() == [1.0] * 4


def test_saved_models_reload_and_average(tmp_path):
    prep = make_prep(n_seeds=2)
    torch.manual_seed(0)
    model = SequenceModel(len(prep["row_columns"]), length=prep["history_length"])
    for seed in prep["seeds"]:  # two identical seeds: their average equals one model
        torch.save(model.state_dict(), tmp_path / f"sequence_seed{seed}.pt")
    models = load_models(tmp_path, prep)
    assert all(not m.training for m in models)  # dropout off for scoring

    rng = np.random.default_rng(1)
    n = 50
    sender, receiver = rng.integers(0, 10, n), rng.integers(0, 10, n)
    minutes = np.sort(rng.integers(0, 500, n))
    events = build_events(sender, receiver, minutes, sender == receiver)
    tx_values = rng.normal(size=(n, 3)).astype(np.float32)
    row_x = encode_rows(make_features(n), prep)

    scores = score_rows(models, prep, events, tx_values, row_x, sender, receiver, minutes,
                        batch_size=16)
    single = score_rows(models[:1], prep, events, tx_values, row_x, sender, receiver, minutes)
    assert scores.shape == (n,)
    assert ((scores > 0) & (scores < 1)).all()
    assert np.allclose(scores, single, atol=1e-6)
    again = score_rows(models, prep, events, tx_values, row_x, sender, receiver, minutes,
                       batch_size=16)
    assert np.array_equal(scores, again)  # same input, same score: dropout is off