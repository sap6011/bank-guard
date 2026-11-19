import numpy as np
import pytest

from bankguard.features import BIG
from bankguard.sequence import build_events, gather, history_bounds

# (sender, receiver, minute, self_transfer); accounts 0, 1 and 2
TXS = [
    (0, 1, 10, False),
    (1, 0, 20, False),
    (0, 0, 30, True),
    (0, 2, 30, False),
    (2, 0, 40, False),
    (0, 1, 50, False),
]
SENDER, RECEIVER, MINUTES, SELF = (np.array(col) for col in zip(*TXS, strict=True))
EVENTS = build_events(SENDER, RECEIVER, MINUTES, SELF)
# per transaction: log_amount, amount_vs_currency, self_transfer
TX_VALUES = np.column_stack([np.arange(6.0), np.arange(6.0) * 10, SELF.astype(float)])


def history_tx(account, minute, length):
    start, end = history_bounds(EVENTS, [account], [minute], length)
    return EVENTS.tx[start[0]:end[0]].tolist()


def test_one_event_per_side_and_self_transfer_once():
    assert len(EVENTS.keys) == 6 + 5


def test_history_is_strictly_earlier():
    # the self transfer and the payment to account 2 happen at minute 30: not included
    assert history_tx(0, 30, 64) == [0, 1]
    assert history_tx(0, 10, 64) == []


def test_length_keeps_the_most_recent_events_oldest_first():
    assert history_tx(0, 50, 64) == [0, 1, 2, 3, 4]
    assert history_tx(0, 50, 3) == [2, 3, 4]


def test_gather_fields_and_front_padding():
    start, end = history_bounds(EVENTS, [0], [50], 4)
    values, mask = gather(EVENTS, TX_VALUES, start, end, [50], [1], 4)
    assert mask.tolist() == [[True, True, True, True]]
    # last event: account 0 received transaction 4 from account 2, ten minutes earlier
    assert values[0, -1].tolist() == pytest.approx([4, 40, 0, np.log1p(10 / 60), 0, 0])
    # first event: account 0 received transaction 1 from account 1, the current counterparty
    assert values[0, 0].tolist() == pytest.approx([1, 10, 0, np.log1p(30 / 60), 0, 1])

    start, end = history_bounds(EVENTS, [0], [30], 4)
    values, mask = gather(EVENTS, TX_VALUES, start, end, [30], [2], 4)
    assert mask.tolist() == [[False, False, True, True]]
    assert not values[0, :2].any()


def test_no_later_event_ever_enters_a_history():
    rng = np.random.default_rng(0)
    n, accounts, length = 2_000, 50, 8
    sender = rng.integers(0, accounts, n)
    receiver = rng.integers(0, accounts, n)
    minutes = np.sort(rng.integers(0, 1_000, n))
    events = build_events(sender, receiver, minutes, sender == receiver)

    start, end = history_bounds(events, sender, minutes, length)
    for i in range(n):
        keys = events.keys[start[i]:end[i]]
        assert (keys // BIG == sender[i]).all()
        assert (keys % BIG < minutes[i]).all()
        earlier = (events.keys // BIG == sender[i]) & (events.keys % BIG < minutes[i])
        assert end[i] - start[i] == min(length, earlier.sum())