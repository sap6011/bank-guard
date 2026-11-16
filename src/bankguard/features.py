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

def minutes_from_start(tx: pd.DataFrame) -> np.ndarray:
    """Whole minutes since the earliest transaction."""
    delta = tx["timestamp"] - tx["timestamp"].min()
    return (delta.dt.total_seconds() // 60).to_numpy().astype(np.int64)


def add_graph_features(tx: pd.DataFrame, windows: dict = WINDOWS) -> pd.DataFrame:
    """Trailing activity counts for sender, receiver and the pair.

    Must be given the full table, not one split: validation rows need
    the train rows before them as history. Self transfers are scored
    but never counted as events.
    """
    from_code, to_code = account_codes(tx)
    minutes = minutes_from_start(tx)

    real = ~tx["self_transfer"].to_numpy()
    ev_from, ev_to, ev_min = from_code[real], to_code[real], minutes[real]

    out = pd.DataFrame(index=tx.index)
    for label, w in windows.items():
        out[f"sender_out_{label}"] = count_prior(ev_from, ev_min, from_code, minutes, w)
        out[f"sender_in_{label}"] = count_prior(ev_to, ev_min, from_code, minutes, w)
        out[f"receiver_in_{label}"] = count_prior(ev_to, ev_min, to_code, minutes, w)
        out[f"receiver_out_{label}"] = count_prior(ev_from, ev_min, to_code, minutes, w)

    n_codes = int(max(from_code.max(), to_code.max())) + 1
    _, pair_code = np.unique(from_code.astype(np.int64) * n_codes + to_code, return_inverse=True)
    out["pair_prior_3d"] = count_prior(pair_code[real], ev_min, pair_code, minutes, WINDOWS["3d"])
    return out

def add_breadth_features(tx: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    """How many *new* counterparties each side has dealt with in the last 3 days.

    `graph` is the output of add_graph_features on the same table.
    """
    from_code, to_code = account_codes(tx)
    minutes = minutes_from_start(tx)
    real = ~tx["self_transfer"].to_numpy()
    first_contact = real & (graph["pair_prior_3d"].to_numpy() == 0)

    out = pd.DataFrame(index=tx.index)
    out["sender_new_pair_out_3d"] = count_prior(
        from_code[first_contact], minutes[first_contact], from_code, minutes, WINDOWS["3d"])
    out["receiver_new_pair_in_3d"] = count_prior(
        to_code[first_contact], minutes[first_contact], to_code, minutes, WINDOWS["3d"])
    return out

def add_attribute_features(tx: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Owner attributes for the sender and receiver of each transaction.

    `accounts` is the output of data.load_accounts. Accounts missing from
    it get missing values; no transaction row is ever dropped or duplicated.
    """
    from_code, to_code = account_codes(tx)
    n_codes = int(max(from_code.max(), to_code.max())) + 1

    # One row per account code that appears in the transactions.
    seen = pd.concat([
        pd.DataFrame({"code": from_code, "bank_id": tx["from_bank"].to_numpy(),
                      "account": tx["from_account"].astype(str).to_numpy()}),
        pd.DataFrame({"code": to_code, "bank_id": tx["to_bank"].to_numpy(),
                      "account": tx["to_account"].astype(str).to_numpy()}),
    ]).drop_duplicates("code")
    by_code = (
        seen.merge(accounts, on=["bank_id", "account"], how="left")
        .set_index("code")
        .reindex(range(n_codes))
    )

    out = pd.DataFrame(index=tx.index)
    for side, code in [("sender", from_code), ("receiver", to_code)]:
        out[f"{side}_entity_type"] = by_code["entity_type"].to_numpy()[code]
        out[f"{side}_bank_country"] = by_code["bank_country"].to_numpy()[code]
        out[f"{side}_entity_accounts"] = by_code["entity_accounts"].to_numpy()[code]
    entity = by_code["entity_id"].to_numpy()
    out["same_entity"] = (entity[from_code] == entity[to_code]).astype(int)
    out["same_bank"] = (tx["from_bank"].to_numpy() == tx["to_bank"].to_numpy()).astype(int)
    return out