"""The sequence transformer: the model, its input preparation and scoring.

Needs torch, which is in the optional "deep" dependency group.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from bankguard.features import BREADTH, CATEGORICAL, FEATURES, GRAPH
from bankguard.model import gbm_frame
from bankguard.sequence import Events, gather, history_bounds

COUNTS = GRAPH + BREADTH + ["sender_entity_accounts", "receiver_entity_accounts"]


class HistoryEncoder(nn.Module):
    """A small transformer over one side's history, read out through a summary token."""

    def __init__(self, n_fields=6, width=64, heads=4, layers=2, dropout=0.1, length=64):
        super().__init__()
        self.embed = nn.Linear(n_fields, width)
        self.summary = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.position = nn.Parameter(torch.randn(1, length + 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, values, mask):
        n = len(values)
        x = torch.cat([self.summary.expand(n, -1, -1), self.embed(values)], dim=1) + self.position
        keep = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=~keep)
        return self.norm(x[:, 0])


class SequenceModel(nn.Module):
    """The current row's features plus the sender's and the receiver's history."""

    def __init__(self, n_row, width=64, dropout=0.1, length=64):
        super().__init__()
        self.row = nn.Sequential(nn.Linear(n_row, width), nn.GELU())
        self.sender = HistoryEncoder(width=width, dropout=dropout, length=length)
        self.receiver = HistoryEncoder(width=width, dropout=dropout, length=length)
        self.head = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1)
        )

    def forward(self, row, sender_values, sender_mask, receiver_values, receiver_mask):
        parts = [
            self.row(row),
            self.sender(sender_values, sender_mask),
            self.receiver(receiver_values, receiver_mask),
        ]
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def load_preprocessing(path) -> dict:
    """The input preparation values saved from training."""
    return json.loads(Path(path).read_text())


def encode_rows(X: pd.DataFrame, prep: dict) -> np.ndarray:
    """The 22 features as the 32 standardised inputs used in training."""
    frame = gbm_frame(X, FEATURES, prep["categories"])
    numeric = frame.drop(columns=CATEGORICAL).astype("float32")
    numeric[COUNTS] = np.log1p(numeric[COUNTS])
    onehot = pd.get_dummies(frame[CATEGORICAL], dtype="float32")
    rows = pd.concat([numeric, onehot], axis=1).reindex(columns=prep["row_columns"]).fillna(0)
    mean = np.array(prep["row_mean"], np.float32)
    std = np.array(prep["row_std"], np.float32)
    return (rows.to_numpy(np.float32) - mean) / std


def standardise_history(values, mask, prep: dict) -> np.ndarray:
    """Saved or gathered histories with the two amount fields scaled as in training.

    Padding stays at zero. Returns a new array; the input is not changed.
    """
    values = values.copy()
    mean = np.array(prep["amount_mean"], np.float32)
    std = np.array(prep["amount_std"], np.float32)
    values[..., :2] = (values[..., :2] - mean) / std * mask[..., None]
    return values


def history_inputs(events: Events, tx_values, start, end, minute, counterparty, prep: dict):
    """One side's histories from the event table, standardised for the model."""
    values, mask = gather(events, tx_values, start, end, minute, counterparty, prep["history_length"])
    return standardise_history(values, mask, prep), mask


def load_models(directory, prep: dict, device="cpu") -> list[SequenceModel]:
    """The trained model of each seed, ready for scoring."""
    models = []
    for seed in prep["seeds"]:
        model = SequenceModel(len(prep["row_columns"]), length=prep["history_length"])
        weights = torch.load(Path(directory) / f"sequence_seed{seed}.pt", map_location=device)
        model.load_state_dict(weights)
        models.append(model.to(device).eval())
    return models


@torch.no_grad()
def predict(models, row_x, sender_values, sender_mask, receiver_values, receiver_mask,
            device="cpu", batch_size=2048) -> np.ndarray:
    """Laundering score for prepared inputs, averaged over the seed models."""
    arrays = (row_x, sender_values, sender_mask, receiver_values, receiver_mask)
    out = []
    for i in range(0, len(row_x), batch_size):
        inputs = [torch.from_numpy(a[i:i + batch_size]).to(device) for a in arrays]
        probs = torch.stack([torch.sigmoid(m(*inputs)) for m in models])
        out.append(probs.mean(dim=0).cpu().numpy())
    return np.concatenate(out)


def score_rows(models, prep: dict, events: Events, tx_values, row_x, sender, receiver, minute,
               device="cpu", batch_size=2048) -> np.ndarray:
    """Laundering score for each row, histories built from the event table.

    The scores rank transfers well but are not probabilities: training kept
    every laundering row and only a sample of normal ones.
    """
    sender, receiver, minute = (np.asarray(a) for a in (sender, receiver, minute))
    length = prep["history_length"]
    s0, s1 = history_bounds(events, sender, minute, length)
    r0, r1 = history_bounds(events, receiver, minute, length)
    out = []
    for i in range(0, len(row_x), batch_size):
        b = slice(i, i + batch_size)
        sv, sm = history_inputs(events, tx_values, s0[b], s1[b], minute[b], receiver[b], prep)
        rv, rm = history_inputs(events, tx_values, r0[b], r1[b], minute[b], sender[b], prep)
        out.append(predict(models, row_x[b], sv, sm, rv, rm, device, batch_size))
    return np.concatenate(out)
