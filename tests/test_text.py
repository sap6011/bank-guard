import numpy as np
import pandas as pd

from bankguard.text import to_text


def test_to_text_writes_every_feature_in_a_fixed_format():
    row = pd.DataFrame([{
        "log_amount": 5.61, "amount_vs_currency": 2.64,
        "hour_sin": np.sin(2 * np.pi * 14 / 24), "hour_cos": np.cos(2 * np.pi * 14 / 24),
        "self_transfer": 0,
        "sender_out_1d": 1662, "sender_in_1d": 1, "sender_out_3d": 1662, "sender_in_3d": 1,
        "sender_new_pair_out_3d": 1001,
        "receiver_in_1d": 2, "receiver_out_1d": 0, "receiver_in_3d": 2, "receiver_out_3d": 0,
        "receiver_new_pair_in_3d": 1, "pair_prior_3d": 2,
        "sender_entity_type": "Sole Proprietorship", "sender_entity_accounts": 6.0,
        "receiver_entity_type": None, "receiver_entity_accounts": np.nan,
        "same_entity": 0, "same_bank": 1,
    }])
    assert to_text(row).iloc[0] == (
        "amount 5.6 log10, 2.6 vs typical for currency. hour 14. "
        "sender: 1662 out, 1 in last day; 1662 out, 1 in last 3 days; 1001 new payees. "
        "receiver: 2 in, 0 out last day; 2 in, 0 out last 3 days; 1 new payers. "
        "pair seen 2 times. sender Sole Proprietorship with 6 accounts, "
        "receiver unknown with unknown accounts. same owner no, same bank yes."
    )