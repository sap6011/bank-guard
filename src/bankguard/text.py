"""Transactions written as sentences, for the text models."""

import numpy as np
import pandas as pd


def _int(s: pd.Series, missing: str = "unknown") -> pd.Series:
    """Whole numbers as text, with missing values spelled out."""
    out = s.round().astype("Int64").astype(str)
    return out.where(s.notna(), missing)


def to_text(X: pd.DataFrame) -> pd.Series:
    """Write each row of the 22 features as one short sentence."""
    hour = (np.round(np.arctan2(X["hour_sin"], X["hour_cos"]) * 12 / np.pi) % 24).astype(int)

    def yes_no(s):
        return np.where(s == 1, "yes", "no")

    return (
        "amount " + X["log_amount"].round(1).astype(str)
        + " log10, " + X["amount_vs_currency"].round(1).astype(str) + " vs typical for currency. "
        + "hour " + hour.astype(str) + ". "
        + np.where(X["self_transfer"] == 1, "self transfer. ", "")
        + "sender: " + _int(X["sender_out_1d"]) + " out, " + _int(X["sender_in_1d"]) + " in last day; "
        + _int(X["sender_out_3d"]) + " out, " + _int(X["sender_in_3d"]) + " in last 3 days; "
        + _int(X["sender_new_pair_out_3d"]) + " new payees. "
        + "receiver: " + _int(X["receiver_in_1d"]) + " in, " + _int(X["receiver_out_1d"]) + " out last day; "
        + _int(X["receiver_in_3d"]) + " in, " + _int(X["receiver_out_3d"]) + " out last 3 days; "
        + _int(X["receiver_new_pair_in_3d"]) + " new payers. "
        + "pair seen " + _int(X["pair_prior_3d"]) + " times. "
        + "sender " + X["sender_entity_type"].fillna("unknown").astype(str)
        + " with " + _int(X["sender_entity_accounts"]) + " accounts, "
        + "receiver " + X["receiver_entity_type"].fillna("unknown").astype(str)
        + " with " + _int(X["receiver_entity_accounts"]) + " accounts. "
        + "same owner " + yes_no(X["same_entity"]) + ", same bank " + yes_no(X["same_bank"]) + "."
    )