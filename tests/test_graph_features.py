import numpy as np
import pandas as pd

from bankguard.features import add_breadth_features, add_graph_features


def make_tx(rows):
    """rows: (timestamp, from_bank, from_account, to_bank, to_account)"""
    tx = pd.DataFrame(rows, columns=["timestamp", "from_bank", "from_account",
                                     "to_bank", "to_account"])
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx["self_transfer"] = (tx["from_account"] == tx["to_account"]) & (tx["from_bank"] == tx["to_bank"])
    return tx


def random_tx(n=400, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-09-01")
    accounts = list("ABCDEFGH")
    rows = [
        (start + pd.Timedelta(minutes=int(rng.integers(0, 6 * 1440))),
         int(rng.integers(1, 3)), rng.choice(accounts),
         int(rng.integers(1, 3)), rng.choice(accounts))
        for _ in range(n)
    ]
    return make_tx(rows)


SMALL = make_tx([
    ("2022-09-01 00:00", 10, "A", 10, "B"),  # 0: first ever transfer
    ("2022-09-01 01:00", 10, "A", 10, "B"),  # 1: same pair again
    ("2022-09-01 01:00", 10, "A", 10, "C"),  # 2: same minute as row 1
    ("2022-09-01 02:00", 10, "A", 10, "A"),  # 3: self transfer
    ("2022-09-01 03:00", 10, "B", 10, "A"),  # 4: B pays A back
    ("2022-09-05 00:00", 10, "A", 10, "B"),  # 5: four days later
])


def test_counts_on_a_hand_worked_example():
    G = add_graph_features(SMALL)
    # Row 2 does not count row 1 (same minute). Row 5 is outside both windows.
    assert G["sender_out_1d"].tolist() == [0, 1, 1, 3, 0, 0]
    assert G["sender_out_3d"].tolist() == [0, 1, 1, 3, 0, 0]
    assert G["pair_prior_3d"].tolist() == [0, 1, 0, 0, 0, 0]
    # Row 4: sender B received rows 0 and 1; receiver A sent rows 0, 1 and 2.
    assert G.loc[4, "sender_in_1d"] == 2
    assert G.loc[4, "receiver_out_1d"] == 3


def test_self_transfers_are_never_counted_as_events():
    G = add_graph_features(SMALL)
    # Row 4 pays A after A's self transfer (row 3). A has received nothing real.
    assert G.loc[4, "receiver_in_1d"] == 0


def test_features_do_not_use_the_future():
    tx = random_tx()
    cutoff = pd.Timestamp("2022-09-04")
    past = tx[tx["timestamp"] < cutoff]
    full = add_graph_features(tx).loc[past.index]
    alone = add_graph_features(past)
    # Deleting every later transaction must not change any earlier row.
    pd.testing.assert_frame_equal(full, alone)


def test_row_order_does_not_matter():
    tx = random_tx()
    shuffled = tx.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(add_graph_features(tx), add_graph_features(shuffled).loc[tx.index])


FAN = make_tx([
    ("2022-09-01 00:00", 10, "A", 10, "B"),  # 0: A's first new counterparty
    ("2022-09-01 01:00", 10, "A", 10, "C"),  # 1: second
    ("2022-09-01 02:00", 10, "A", 10, "D"),  # 2: third
    ("2022-09-01 03:00", 10, "A", 10, "B"),  # 3: B again, not new
    ("2022-09-01 04:00", 10, "A", 10, "E"),  # 4: fourth new one
    ("2022-09-01 05:00", 10, "F", 10, "E"),  # 5: E now paid by a second sender
])


def test_breadth_counts_only_new_counterparties():
    B = add_breadth_features(FAN, add_graph_features(FAN))
    # Row 4 still sees 3, because the repeat payment to B (row 3) is not a new pair.
    assert B["sender_new_pair_out_3d"].tolist() == [0, 1, 2, 3, 3, 0]
    assert B["receiver_new_pair_in_3d"].tolist() == [0, 0, 0, 1, 0, 1]


def test_breadth_does_not_use_the_future():
    tx = random_tx()
    past = tx[tx["timestamp"] < pd.Timestamp("2022-09-04")]
    full = add_breadth_features(tx, add_graph_features(tx)).loc[past.index]
    alone = add_breadth_features(past, add_graph_features(past))
    pd.testing.assert_frame_equal(full, alone)