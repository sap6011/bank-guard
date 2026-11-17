import pytest

from bankguard.data import NO_PATTERN, load_patterns, load_transactions, typology_for

PATTERNS = """BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out
2022/09/01 00:06,010,A,020,B,100.00,Euro,100.00,Euro,ACH,1
2022/09/01 04:33,010,A,030,C,200.00,Euro,200.00,Euro,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops
2022/09/02 08:44,020,B,010,A,50.00,Euro,50.00,Euro,ACH,1
END LAUNDERING ATTEMPT - CYCLE
"""

TX = """Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:06,10,A,20,B,100.00,Euro,100.00,Euro,ACH,1
2022/09/01 04:33,10,A,30,C,200.00,Euro,200.00,Euro,ACH,1
2022/09/02 08:44,20,B,10,A,50.00,Euro,50.00,Euro,ACH,1
2022/09/02 09:00,20,B,10,A,75.00,Euro,75.00,Euro,ACH,1
"""


@pytest.fixture
def files(tmp_path):
    (tmp_path / "patterns.txt").write_text(PATTERNS)
    (tmp_path / "tx.csv").write_text(TX)
    return tmp_path


def test_load_patterns_reads_attempts_and_typologies(files):
    p = load_patterns(files / "patterns.txt")
    assert len(p) == 3
    assert p["typology"].tolist() == ["FAN-OUT", "FAN-OUT", "CYCLE"]
    assert p["attempt_id"].tolist() == [0, 0, 1]


def test_typology_matches_despite_leading_zeros_in_bank_ids(files):
    # The patterns file writes bank 10 as "010"; the match must still work.
    tx = load_transactions(files / "tx.csv")
    typology = typology_for(tx, load_patterns(files / "patterns.txt"))
    assert typology.tolist() == ["FAN-OUT", "FAN-OUT", "CYCLE", NO_PATTERN]