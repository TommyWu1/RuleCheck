import pytest

from policy_engine import vacation_days


@pytest.mark.parametrize(
    ("months_employed", "expected_days"),
    [(0, 0), (11, 0), (12, 10), (13, 10), (59, 10), (60, 15), (61, 15)],
)
def test_vacation_days_at_policy_boundaries(months_employed, expected_days):
    assert vacation_days(months_employed) == expected_days


def test_vacation_days_rejects_negative_tenure():
    with pytest.raises(ValueError, match="months_employed cannot be negative"):
        vacation_days(-1)
