"""Feature building for the laundering model."""

import numpy as np
import pandas as pd

NUMERIC = ["log_amount", "amount_vs_currency", "hour_sin", "hour_cos", "self_transfer"]
SHORTCUTS = ["payment_format", "payment_currency"]


def fit_currency_median(train: pd.DataFrame) -> dict[str, float]:
    """Typical payment size per currency, learned from the training split only."""
    med = train.groupby("payment_currency", observed=True)["amount_paid"].median()
    return {str(k): float(v) for k, v in med.items()}


def add_features(df: pd.DataFrame, currency_median: dict[str, float]) -> pd.DataFrame:
    """Per-transaction features. Uses nothing but the row itself and currency_median."""
    out = pd.DataFrame(index=df.index)
    out["log_amount"] = np.log10(df["amount_paid"].clip(lower=0.01))
    typical = df["payment_currency"].astype(str).map(currency_median)
    out["amount_vs_currency"] = out["log_amount"] - np.log10(typical)
    hour = df["timestamp"].dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["self_transfer"] = df["self_transfer"].astype(int)
    out["payment_format"] = df["payment_format"].astype(str)
    out["payment_currency"] = df["payment_currency"].astype(str)
    return out

BIG = 100_000  # must exceed the number of minutes in the dataset (18 days = 25,920)
WINDOWS = {"1d": 1440, "3d": 4320}  # window lengths in minutes


def account_codes(tx: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Give every (bank, account) one integer id, shared by the sender and receiver side."""
    from_key = tx["from_bank"].astype(str) + "_" + tx["from_account"].astype(str)
    to_key = tx["to_bank"].astype(str) + "_" + tx["to_account"].astype(str)
    codes, _ = pd.factorize(pd.concat([from_key, to_key], ignore_index=True))
    return codes[: len(tx)], codes[len(tx):]


def count_prior(event_codes, event_minutes, query_codes, query_minutes, window):
    """For each query, count events for the same code in [t - window, t).

    Strictly earlier than t: an event in the same minute is not counted.
    """
    events = np.sort(event_codes.astype(np.int64) * BIG + event_minutes)
    q = query_codes.astype(np.int64) * BIG + query_minutes
    return np.searchsorted(events, q, "left") - np.searchsorted(events, q - window, "left")