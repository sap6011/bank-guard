"""Loading and splitting the IBM AML HI-Small transactions."""

from pathlib import Path

import numpy as np
import pandas as pd

TX_COLS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid",
    "payment_currency", "payment_format", "is_laundering",
]

TX_DTYPES = {
    "from_bank": "int32", "from_account": "category",
    "to_bank": "int32", "to_account": "category",
    "amount_received": "float64", "receiving_currency": "category",
    "amount_paid": "float64", "payment_currency": "category",
    "payment_format": "category", "is_laundering": "int8",
}

# Inclusive date ranges. 11 to 18 Sep is the thin tail, deliberately left out.
SPLITS = {
    "train": ("2022-09-01", "2022-09-06"),
    "validation": ("2022-09-07", "2022-09-08"),
    "test": ("2022-09-09", "2022-09-10"),
}


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read the transactions file and add `date` and `self_transfer`.

    The raw file has two columns both named "Account", so the header
    row is replaced with our own names.
    """
    tx = pd.read_csv(path, header=0, names=TX_COLS, dtype=TX_DTYPES)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], format="%Y/%m/%d %H:%M")
    tx["date"] = tx["timestamp"].dt.normalize()
    # Account ids are only unique within a bank, so both must match.
    tx["self_transfer"] = (
        (tx["from_account"].astype(str) == tx["to_account"].astype(str))
        & (tx["from_bank"] == tx["to_bank"])
    )
    return tx


def split_by_date(tx: pd.DataFrame, splits: dict = SPLITS) -> dict[str, pd.DataFrame]:
    """Split by calendar date so validation and test are always later than train."""
    return {
        name: tx[(tx["date"] >= start) & (tx["date"] <= end)]
        for name, (start, end) in splits.items()
    }

ACCOUNT_COLS = ["bank_name", "bank_id", "account", "entity_id", "entity_name"]


def load_accounts(path: str | Path) -> pd.DataFrame:
    """Read the accounts file and derive owner attributes, one row per (bank, account)."""
    accounts = pd.read_csv(path, header=0, names=ACCOUNT_COLS)
    # "Sole Proprietorship #123" -> "Sole Proprietorship"
    accounts["entity_type"] = accounts["entity_name"].str.replace(r"\s*#\d+$", "", regex=True)
    # "Germany Bank #45" -> "Germany". Names that do not fit the pattern become "Other".
    accounts["bank_country"] = (
        accounts["bank_name"].str.extract(r"^(.+?) Bank #\d+$")[0].fillna("Other")
    )
    # How many accounts the same owner holds. Counted before removing duplicates.
    accounts["entity_accounts"] = accounts.groupby("entity_id")["account"].transform("size")
    return accounts.drop_duplicates(["bank_id", "account"])

# Columns that identify one transaction when matching the patterns file.
PATTERN_KEY = ["timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_paid"]
NO_PATTERN = "No listed pattern"


def load_patterns(path: str | Path) -> pd.DataFrame:
    """Read the patterns file: one row per transaction in a listed laundering attempt."""
    rows, attempt_id, typology = [], -1, None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt_id += 1
                # "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out" -> "FAN-OUT"
                typology = line.split(" - ", 1)[1].split(":")[0].strip()
            elif line and not line.startswith("END LAUNDERING ATTEMPT"):
                rows.append(line.split(",") + [attempt_id, typology])
    p = pd.DataFrame(rows, columns=TX_COLS + ["attempt_id", "typology"])
    p["timestamp"] = pd.to_datetime(p["timestamp"], format="%Y/%m/%d %H:%M")
    p["from_bank"] = p["from_bank"].astype("int32")
    p["to_bank"] = p["to_bank"].astype("int32")
    p["amount_paid"] = p["amount_paid"].astype(float)
    return p.drop_duplicates(PATTERN_KEY)[PATTERN_KEY + ["attempt_id", "typology"]]


def typology_for(tx: pd.DataFrame, patterns: pd.DataFrame) -> np.ndarray:
    """The laundering typology of each transaction, or NO_PATTERN."""
    key = tx[PATTERN_KEY].copy()
    key["from_account"] = key["from_account"].astype(str)
    key["to_account"] = key["to_account"].astype(str)
    merged = key.merge(patterns, on=PATTERN_KEY, how="left")
    if len(merged) != len(tx):
        raise ValueError("pattern rows matched more than one transaction")
    return merged["typology"].fillna(NO_PATTERN).to_numpy()