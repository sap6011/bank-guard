import numpy as np
import pandas as pd
import pytest

from bankguard.features import add_features, fit_currency_median


def make(rows):
    """rows: (timestamp, amount_paid, currency, format, self_transfer)"""
    df = pd.DataFrame(rows, columns=["timestamp", "amount_paid", "payment_currency",
                                     "payment_format", "self_transfer"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


TRAIN = make([
    ("2022-09-01 00:00", 10.0, "US Dollar", "ACH", False),
    ("2022-09-01 06:00", 100.0, "US Dollar", "ACH", False),
    ("2022-09-02 12:00", 1000.0, "US Dollar", "Cheque", True),
    ("2022-09-02 18:00", 50.0, "Euro", "ACH", False),
])


def test_currency_median_per_currency():
    assert fit_currency_median(TRAIN) == {"US Dollar": 100.0, "Euro": 50.0}


def test_amount_features():
    X = add_features(TRAIN, fit_currency_median(TRAIN))
    assert X["log_amount"].tolist() == pytest.approx([1.0, 2.0, 3.0, np.log10(50)])
    # 10x below, equal to, and 10x above the typical US Dollar payment
    assert X["amount_vs_currency"].tolist() == pytest.approx([-1.0, 0.0, 1.0, 0.0])


def test_hour_is_cyclical():
    X = add_features(TRAIN, fit_currency_median(TRAIN))
    assert X["hour_sin"].tolist() == pytest.approx([0, 1, 0, -1], abs=1e-9)
    assert X["hour_cos"].tolist() == pytest.approx([1, 0, -1, 0], abs=1e-9)


def test_medians_come_from_train_not_from_the_data_being_scored():
    medians = fit_currency_median(TRAIN)
    later = make([
        ("2022-09-07 00:00", 100.0, "US Dollar", "ACH", False),
        ("2022-09-07 00:00", 5.0, "Yen", "ACH", False),
    ])
    X = add_features(later, medians)
    assert X["amount_vs_currency"].iloc[0] == pytest.approx(0.0)
    # A currency never seen in train has no typical size: missing, not guessed.
    assert np.isnan(X["amount_vs_currency"].iloc[1])

def test_feature_contract_has_22_columns_and_no_shortcuts():
    from bankguard.features import FEATURES

    assert len(FEATURES) == 22
    assert len(set(FEATURES)) == 22
    for shortcut in ["payment_format", "payment_currency", "sender_bank_country", "receiver_bank_country"]:
        assert shortcut not in FEATURES