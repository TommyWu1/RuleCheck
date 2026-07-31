from copy import deepcopy

import pytest
from pydantic import ValidationError

from models import GeneratedSuite


VALID_PAYLOAD = {
    "cases": [
        {
            "name": "New employee",
            "months_employed": 0,
            "expected_days": 0,
            "rationale": "Fewer than 12 completed months.",
        },
        {
            "name": "Below first threshold",
            "months_employed": 11,
            "expected_days": 0,
            "rationale": "Immediately below 12 completed months.",
        },
        {
            "name": "At first threshold",
            "months_employed": 12,
            "expected_days": 10,
            "rationale": "The 10-day tier starts at 12 months.",
        },
        {
            "name": "Below second threshold",
            "months_employed": 59,
            "expected_days": 10,
            "rationale": "Immediately below 60 completed months.",
        },
        {
            "name": "At second threshold",
            "months_employed": 60,
            "expected_days": 15,
            "rationale": "The 15-day tier starts at 60 months.",
        },
    ]
}


def test_generated_suite_accepts_exactly_five_valid_cases():
    suite = GeneratedSuite.model_validate(VALID_PAYLOAD)

    assert [case.months_employed for case in suite.cases] == [0, 11, 12, 59, 60]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["cases"][0].pop("rationale"),
        lambda payload: payload["cases"][0].update({"source": "invented"}),
        lambda payload: payload["cases"][0].update({"months_employed": -1}),
        lambda payload: payload["cases"].pop(),
    ],
    ids=["missing-field", "extra-field", "negative-tenure", "wrong-case-count"],
)
def test_generated_suite_rejects_malformed_model_output(mutate):
    payload = deepcopy(VALID_PAYLOAD)
    mutate(payload)

    with pytest.raises(ValidationError):
        GeneratedSuite.model_validate(payload)
