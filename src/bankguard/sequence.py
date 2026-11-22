"""Transaction histories for the sequence model.

Every transaction is an event for its sender and one for its receiver. The
history of a query (an account at a minute) is that account's most recent
events strictly before the minute, oldest first.
"""

from dataclasses import dataclass

import numpy as np

from bankguard.features import BIG

EVENT_FIELDS = [
    "log_amount", "amount_vs_currency", "sent",
    "log_hours_before", "self_transfer", "same_counterparty",
]


@dataclass
class Events:
    """Every event, sorted by account and then by time."""

    keys: np.ndarray  # account * BIG + minute, sorted
    counterparty: np.ndarray  # the other account in the transaction
    sent: np.ndarray  # 1 if this account sent the money, 0 if it received it
    tx: np.ndarray  # row position of the transaction


def build_events(sender, receiver, minutes, self_transfer) -> Events:
    """One event per side of each transaction. A self transfer is one event."""
    sender, receiver, minutes = (np.asarray(a, np.int64) for a in (sender, receiver, minutes))
    other = ~np.asarray(self_transfer, bool)
    account = np.concatenate([sender, receiver[other]])
    minute = np.concatenate([minutes, minutes[other]])
    counterparty = np.concatenate([receiver, sender[other]])
    sent = np.concatenate([np.ones(len(sender), np.int8), np.zeros(other.sum(), np.int8)])
    tx = np.concatenate([np.arange(len(sender)), np.flatnonzero(other)])
    keys = account * BIG + minute
    order = np.argsort(keys, kind="stable")
    return Events(keys[order], counterparty[order], sent[order], tx[order])


def history_bounds(events: Events, account, minute, length: int):
    """Start and end positions in `events` of each query's history.

    A history holds at most `length` events of the same account, all
    strictly earlier than the query's minute.
    """
    account = np.asarray(account, np.int64)
    end = np.searchsorted(events.keys, account * BIG + np.asarray(minute, np.int64), "left")
    first = np.searchsorted(events.keys, account * BIG, "left")
    return np.maximum(end - length, first), end


def gather(events: Events, tx_values, start, end, minute, counterparty, length: int):
    """The six event fields for each query, padded at the front to `length`.

    tx_values: per transaction (log_amount, amount_vs_currency, self_transfer).
    Returns values of shape (n, length, 6) and a mask that is True for real events.
    """
    steps = np.arange(length)
    pad = length - (end - start)
    mask = steps[None, :] >= pad[:, None]
    idx = np.where(mask, start[:, None] + steps[None, :] - pad[:, None], 0)

    tx = events.tx[idx]
    hours = (np.asarray(minute)[:, None] - events.keys[idx] % BIG) / 60
    values = np.stack([
        tx_values[tx, 0],
        tx_values[tx, 1],
        events.sent[idx],
        np.log1p(hours),
        tx_values[tx, 2],
        events.counterparty[idx] == np.asarray(counterparty)[:, None],
    ], axis=-1).astype(np.float32)
    values[~mask] = 0
    return values, mask

def remove_events(values, mask, slots):
    """One history with the events in `slots` removed, as if they never happened.

    Older events move up to fill the gaps and padding grows at the front.
    values: (length, n_fields); mask: (length,). Returns the new values and
    mask, and for each new slot the old slot it came from (-1 for padding).
    A full history would in reality pull in older events from beyond its
    length; those are not available here, so the history just gets shorter.
    """
    length = len(mask)
    keep = mask.copy()
    keep[np.asarray(slots, dtype=int)] = False
    old = np.flatnonzero(keep)
    source = np.full(length, -1)
    source[length - len(old):] = old
    new_values = np.zeros_like(values)
    new_values[length - len(old):] = values[old]
    return new_values, source >= 0, source