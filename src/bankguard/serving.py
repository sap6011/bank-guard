"""Scoring demo transfers for the API.

Loads the saved sequence transformer and the demo bundle (built by
scripts/build_demo_bundle.py) once, then scores transfers and turns raw
scores into a rank and an alert decision. Needs torch.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bankguard.features import CATEGORICAL
from bankguard.sequence import remove_events
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

    def currency(self, i: int) -> str:
        """Payment currency of demo transfer i."""
        d = self.data
        return str(d["currency_names"][d["detail_currency"][d["tx"][i]]])

    def explain(self, i: int):
        """How much each past transfer raises demo transfer i's score.

        Removes each past transfer in turn, leaving everything else as it is,
        and scores again. Returns the base score and a table with one row per
        past transfer: side, slot, its row in the details table, and
        effect = base score minus the score without it. Largest effect first.
        """
        base = self.score(i)[0]
        variants = [
            (side, int(slot))
            for side in SIDES
            for slot in np.flatnonzero(self.data[f"{side}_mask"][i])
        ]
        columns = ["side", "slot", "detail", "effect"]
        if not variants:
            return base, pd.DataFrame(columns=columns)

        n = len(variants)
        row = np.repeat(self.row_x[i:i + 1], n, axis=0)
        values = {s: np.repeat(self.history[s][i:i + 1], n, axis=0) for s in SIDES}
        masks = {s: np.repeat(self.data[f"{s}_mask"][i:i + 1], n, axis=0) for s in SIDES}
        for k, (side, slot) in enumerate(variants):
            values[side][k], masks[side][k], _ = remove_events(values[side][k], masks[side][k], [slot])
        scores = predict(self.models, row, values["sender"], masks["sender"],
                         values["receiver"], masks["receiver"], self.device)

        table = pd.DataFrame({
            "side": [side for side, _ in variants],
            "slot": [slot for _, slot in variants],
            "detail": [int(self.data[f"{side}_tx"][i, slot]) for side, slot in variants],
            "effect": base - scores,
        })
        return base, table.sort_values("effect", ascending=False, ignore_index=True)

    def what_if(self, i: int, amount: float | None = None, removed=()) -> float:
        """Score demo transfer i with a different amount and/or past transfers removed.

        amount is in the transfer's own payment currency; removed holds
        (side, slot) pairs. Changing the amount changes only the two amount
        features: a transfer is never part of its own history or counts.
        """
        features = self.features.iloc[[i]].copy()
        if amount is not None:
            if amount <= 0:
                raise ValueError("amount must be positive")
            log_amount = np.log10(max(amount, 0.01))
            median = self.prep["currency_median"][self.currency(i)]
            features["log_amount"] = log_amount
            features["amount_vs_currency"] = log_amount - np.log10(median)
        row = encode_rows(features, self.prep)

        if any(side not in SIDES for side, _ in removed):
            raise ValueError(f"side must be one of {SIDES}")
        inputs = []
        for side in SIDES:
            slots = [slot for s, slot in removed if s == side]
            v, m, _ = remove_events(self.history[side][i], self.data[f"{side}_mask"][i], slots)
            inputs += [v[None], m[None]]
        return float(predict(self.models, row, *inputs, self.device)[0])