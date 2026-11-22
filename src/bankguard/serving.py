"""Scoring demo transfers for the API.

Loads the saved sequence transformer and the demo bundle (built by
scripts/build_demo_bundle.py) once, then scores transfers and turns raw
scores into a rank and an alert decision. Needs torch.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bankguard.features import CATEGORICAL
from bankguard.sequence_model import (
    encode_rows,
    load_models,
    load_preprocessing,
    predict,
    standardise_history,
)

SIDES = ("sender", "receiver")


class DemoScorer:
    """The models, the bundle and the prepared inputs, loaded once at startup."""

    def __init__(self, model_dir, bundle_path, device="cpu"):
        self.device = device
        self.prep = load_preprocessing(Path(model_dir) / "preprocessing.json")
        self.models = load_models(model_dir, self.prep, device)
        with np.load(bundle_path, allow_pickle=False) as bundle:
            self.data = {name: bundle[name] for name in bundle.files}

        d = self.data
        self.features = pd.DataFrame(d["features"], columns=d["feature_names"].tolist())
        for c in CATEGORICAL:
            self.features[c] = d[c]
        self.row_x = encode_rows(self.features, self.prep)
        self.history = {
            side: standardise_history(d[f"{side}_values"], d[f"{side}_mask"], self.prep)
            for side in SIDES
        }

    def __len__(self) -> int:
        return len(self.data["tx"])

    def score(self, idx) -> np.ndarray:
        """Raw scores of the demo transfers at positions `idx`."""
        idx = np.atleast_1d(idx)
        return predict(
            self.models, self.row_x[idx],
            self.history["sender"][idx], self.data["sender_mask"][idx],
            self.history["receiver"][idx], self.data["receiver_mask"][idx],
            self.device,
        )

    def percentile(self, scores) -> np.ndarray:
        """Percentage of validation transfers scoring at or below each score."""
        q = self.data["reference_quantiles"]
        below = np.searchsorted(q, np.asarray(scores), side="right") - 1
        return np.clip(below / (len(q) - 1) * 100, 0, 100)

    def alerts(self, scores) -> np.ndarray:
        """Whether each score crosses each alert threshold; columns follow `alert_fpr`."""
        return np.asarray(scores)[:, None] >= self.data["alert_threshold"][None, :]