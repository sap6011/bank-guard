import pandas as pd
import pytest

from bankguard.data import load_transactions, split_by_date

RAW = """Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,10,A,10,A,100.0,US Dollar,100.0,US Dollar,Reinvestment,0
2022/09/07 10:00,10,A,20,A,200.0,US Dollar,200.0,US Dollar,ACH,1
2022/09/09 12:30,10,A,10,B,50.0,Euro,50.0,Euro,Cheque,0
2022/09/12 09:00,30,C,10,A,5.0,Euro,5.0,Euro,ACH,1
"""


@pytest.fixture
def tx(tmp_path):
    path = tmp_path / "tx.csv"
    path.write_text(RAW)
    return load_transactions(path)


def test_columns_and_types(tx):
    assert len(tx) == 4
    assert pd.api.types.is_datetime64_any_dtype(tx["timestamp"])
    assert tx["is_laundering"].tolist() == [0, 1, 0, 1]


def test_self_transfer_needs_same_bank_and_account(tx):
    # Row 2 has the same account id at a different bank: not a self transfer.
    assert tx["self_transfer"].tolist() == [True, False, False, False]


def test_split_by_date_excludes_tail(tx):
    parts = split_by_date(tx)
    assert {k: len(v) for k, v in parts.items()} == {"train": 1, "validation": 1, "test": 1}
    assert parts["train"]["date"].max() < parts["validation"]["date"].min()
    assert parts["validation"]["date"].max() < parts["test"]["date"].min()