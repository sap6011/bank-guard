import numpy as np
import pandas as pd
import pytest

from bankguard.data import load_accounts
from bankguard.features import add_attribute_features

RAW_ACCOUNTS = """Bank Name,Bank ID,Account Number,Entity ID,Entity Name
Germany Bank #1,10,A,E1,Corporation #5
Germany Bank #1,10,B,E1,Corporation #5
France Bank #2,20,A,E2,Sole Proprietorship #9
First Savings,30,C,E3,Partnership #1
"""


@pytest.fixture
def accounts(tmp_path):
    path = tmp_path / "accounts.csv"
    path.write_text(RAW_ACCOUNTS)
    return load_accounts(path)


def make_tx(rows):
    return pd.DataFrame(rows, columns=["from_bank", "from_account", "to_bank", "to_account"])


TX = make_tx([
    (10, "A", 10, "B"),  # 0: same owner, same bank
    (10, "A", 20, "A"),  # 1: same account id, different bank and owner
    (30, "C", 10, "A"),  # 2: bank name without a country
    (10, "A", 99, "Z"),  # 3: receiver missing from the accounts file
])


def test_load_accounts_derives_owner_attributes(accounts):
    a = accounts.set_index(["bank_id", "account"])
    assert a.loc[(10, "A"), "entity_type"] == "Corporation"
    assert a.loc[(20, "A"), "entity_type"] == "Sole Proprietorship"
    assert a.loc[(10, "A"), "bank_country"] == "Germany"
    assert a.loc[(30, "C"), "bank_country"] == "Other"
    assert a.loc[(10, "A"), "entity_accounts"] == 2


def test_attributes_follow_bank_and_account_together(accounts):
    A = add_attribute_features(TX, accounts)
    assert len(A) == len(TX)
    assert A["sender_entity_type"].tolist() == ["Corporation"] * 2 + ["Partnership", "Corporation"]
    # Row 1: account "A" at bank 20 is a different owner from "A" at bank 10.
    assert A.loc[1, "receiver_entity_type"] == "Sole Proprietorship"
    assert A["same_entity"].tolist() == [1, 0, 0, 0]
    assert A["same_bank"].tolist() == [1, 0, 0, 0]


def test_unknown_account_gets_missing_values_not_a_dropped_row(accounts):
    A = add_attribute_features(TX, accounts)
    assert pd.isna(A.loc[3, "receiver_entity_type"])
    assert np.isnan(A.loc[3, "receiver_entity_accounts"])
    assert A.loc[3, "same_entity"] == 0