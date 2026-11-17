import numpy as np
import pytest

from bankguard.data import NO_PATTERN
from bankguard.evaluate import GROUPS, laundering_group, recall_by_group


def test_laundering_group_splits_on_pattern_then_sender_activity():
    typology = ["FAN-OUT", NO_PATTERN, NO_PATTERN, "CYCLE"]
    sender_out_3d = [5000, 5000, 3, 3]
    assert laundering_group(typology, sender_out_3d).tolist() == [
        "patterned", "unpatterned, hub sender", "unpatterned, quiet sender", "patterned",
    ]


def test_recall_by_group_uses_a_threshold_set_on_normal_rows():
    # 100 normal rows scored 0.00 to 0.99; a 1% budget flags only the top one (0.99).
    normal = np.arange(100) / 100
    y = np.r_[np.zeros(100), np.ones(3)]
    scores = np.r_[normal, 0.995, 0.5, 0.995]
    group = np.array([GROUPS[1]] * 100 + [GROUPS[0], GROUPS[1], GROUPS[2]])

    table = recall_by_group(y, scores, group, fprs=(0.01,))
    assert table["laundering"].tolist() == [1, 1, 1]
    # Patterned and hub rows score above the threshold; the quiet sender row does not.
    assert table["recall_at_1.0%_fpr"].tolist() == pytest.approx([1.0, 0.0, 1.0])