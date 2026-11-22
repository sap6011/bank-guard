"""Build the demo bundle that the scoring service loads.

Run once on a machine with the full dataset (a few minutes, several GB of memory):

    uv run python scripts/build_demo_bundle.py

The service cannot rebuild features or histories from the raw 476 MB file, so
this script saves what it needs for a sample of test-period transfers (9 to 10
Sep): each transfer's raw features, its sender's and receiver's last events
exactly as the model saw them in evaluation, and readable details of every
transfer involved, in one compressed NumPy file.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bankguard.data import (
    load_accounts,
    load_patterns,
    load_transactions,
    split_by_date,
    typology_for,
)
from bankguard.evaluate import laundering_group
from bankguard.features import (
    CATEGORICAL,
    FEATURES,
    account_codes,
    build_features,
    minutes_from_start,
)
from bankguard.sequence import build_events, gather, history_bounds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models" / "sequence"
OUT = ROOT / "models" / "demo" / "bundle.npz"
N_TOP_NORMAL = 300  # the normal transfers the model scores highest (false alarms)
N_RANDOM_NORMAL = 3_000  # normal transfers drawn at random
SEED = 0
ALERT_FPRS = [0.01, 0.001]  # alert budgets: share of normal transfers that get flagged
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


def choose_rows(y, scores, rng):
    """Positions within the test split of the demo transfers, in time order, and why each is in."""
    laundering = np.flatnonzero(y == 1)
    normal = np.flatnonzero(y == 0)
    top = normal[np.argsort(scores[normal])[::-1][:N_TOP_NORMAL]]
    random = rng.choice(np.setdiff1d(normal, top), size=N_RANDOM_NORMAL, replace=False)
    chosen = np.concatenate([laundering, top, random])
    reason = np.array(
        ["laundering"] * len(laundering)
        + ["top-scored normal"] * len(top)
        + ["random normal"] * len(random)
    )
    order = np.argsort(chosen)
    return chosen[order], reason[order]


def history_tx(events, start, end, length):
    """Row position of the transaction behind each history slot, -1 for padding.

    Uses the same front padding as sequence.gather, so slot i here is slot i there.
    """
    steps = np.arange(length)
    pad = length - (end - start)
    mask = steps[None, :] >= pad[:, None]
    idx = np.where(mask, start[:, None] + steps[None, :] - pad[:, None], 0)
    return np.where(mask, events.tx[idx], -1)


def main():
    prep = json.loads((MODELS / "preprocessing.json").read_text())
    length = prep["history_length"]

    tx = load_transactions(DATA / "HI-Small_Trans.csv")
    accounts = load_accounts(DATA / "HI-Small_accounts.csv")
    X = build_features(tx, accounts, prep["currency_median"])

    # The demo transfers, chosen with the evaluated test scores.
    splits = split_by_date(tx)
    test_pos = splits["test"].index.to_numpy()
    scores = np.load(MODELS / "sequence_test_scores_avg.npy")
    assert len(scores) == len(test_pos)

    # Alert thresholds and the reference for percentiles come from validation,
    # so nothing shown about a demo score is set on the test days.
    val_pos = splits["validation"].index.to_numpy()
    val_scores = np.load(MODELS / "sequence_val_scores_avg.npy")
    assert len(val_scores) == len(val_pos)
    val_normal = val_scores[tx["is_laundering"].to_numpy()[val_pos] == 0]

    y_test = tx["is_laundering"].to_numpy()[test_pos]
    local, sample = choose_rows(y_test, scores, np.random.default_rng(SEED))
    pos = test_pos[local]

    # Both histories of each demo transfer, built exactly as in evaluation.
    sender, receiver = account_codes(tx)
    minutes = minutes_from_start(tx)
    events = build_events(sender, receiver, minutes, tx["self_transfer"].to_numpy())
    tx_values = X[["log_amount", "amount_vs_currency", "self_transfer"]].to_numpy(np.float32)
    sides = {}
    for side, own, other in [("sender", sender, receiver), ("receiver", receiver, sender)]:
        start, end = history_bounds(events, own[pos], minutes[pos], length)
        values, mask = gather(events, tx_values, start, end, minutes[pos], other[pos], length)
        slots = history_tx(events, start, end, length)
        assert ((slots >= 0) == mask).all()
        sides[side] = (values, mask, slots)

    # Readable details of every transfer involved: the demo transfers and their histories.
    in_history = np.concatenate([slots.ravel() for _, _, slots in sides.values()])
    used = np.unique(np.concatenate([pos, in_history[in_history >= 0]]))

    def row_of(p):
        """Row in the details table for each position; padding (-1) stays -1."""
        return np.where(p >= 0, np.searchsorted(used, p), -1).astype(np.int32)

    d = tx.iloc[used]
    parties = pd.concat([
        d["from_bank"].astype(str) + "/" + d["from_account"].astype(str),
        d["to_bank"].astype(str) + "/" + d["to_account"].astype(str),
    ], ignore_index=True)
    party_codes, account_names = pd.factorize(parties)
    currency_codes, currency_names = pd.factorize(d["payment_currency"].astype(str))
    format_codes, format_names = pd.factorize(d["payment_format"].astype(str))

    typology = typology_for(tx.iloc[pos], load_patterns(DATA / "HI-Small_Patterns.txt"))
    group = laundering_group(typology, X["sender_out_3d"].to_numpy()[pos])
    y = y_test[local]

    bundle = {
        # One entry per demo transfer, in time order.
        "tx": row_of(pos),
        "sample": sample,
        "is_laundering": y.astype(np.int8),
        "typology": np.asarray(typology, dtype=str),
        "group": np.where(y == 1, group, "normal"),
        "evaluated_score": scores[local].astype(np.float32),
        # Turning a raw score into an alert decision and a percentile, set on validation.
        "alert_fpr": np.array(ALERT_FPRS),
        "alert_threshold": np.quantile(val_normal, 1 - np.array(ALERT_FPRS)).astype(np.float32),
        "reference_quantiles": np.quantile(val_scores, np.linspace(0, 1, 10_001)).astype(np.float32),
        "features": X.iloc[pos][NUMERIC].to_numpy(np.float32),
        "feature_names": np.array(NUMERIC),
        **{c: X.iloc[pos][c].astype(str).to_numpy(dtype=str) for c in CATEGORICAL},
        # Both histories, before standardisation, and the transfer behind each slot.
        **{f"{side}_values": v for side, (v, _, _) in sides.items()},
        **{f"{side}_mask": m for side, (_, m, _) in sides.items()},
        **{f"{side}_tx": row_of(s) for side, (_, _, s) in sides.items()},
        # The details table, with repeated text stored once as a list of names.
        "detail_minute": minutes[used].astype(np.int32),
        "time_origin": np.array(str(tx["timestamp"].min())),
        "detail_from": party_codes[: len(used)].astype(np.int32),
        "detail_to": party_codes[len(used):].astype(np.int32),
        "detail_amount": d["amount_paid"].to_numpy(np.float64),
        "detail_currency": currency_codes.astype(np.int16),
        "detail_format": format_codes.astype(np.int16),
        "account_names": np.asarray(account_names, dtype=str),
        "currency_names": np.asarray(currency_names, dtype=str),
        "format_names": np.asarray(format_names, dtype=str),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **bundle)

    # Reload without pickle to prove the file holds plain arrays only.
    with np.load(OUT, allow_pickle=False) as check:
        assert set(check.files) == set(bundle)
    print("demo transfers:", pd.Series(sample).value_counts().to_dict())
    print("laundering groups:", pd.Series(bundle["group"][y == 1]).value_counts().to_dict())
    print(f"transfers in the details table: {len(used):,}")
    print(f"accounts: {len(account_names):,}")
    print(f"bundle: {OUT.stat().st_size / 1e6:.1f} MB at {OUT}")
    for fpr, threshold in zip(ALERT_FPRS, bundle["alert_threshold"]):
        caught = (bundle["evaluated_score"][y == 1] >= threshold).mean()
        print(f"alert budget {fpr:.1%}: threshold {threshold:.4f}, "
              f"flags {caught:.1%} of test laundering")


if __name__ == "__main__":
    main()