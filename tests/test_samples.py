import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.app import Patient

SAMPLES = Path("samples")


@pytest.mark.parametrize("name", ["high_risk", "low_risk"])
def test_valid_samples_match_the_api_schema(name):
    Patient(**json.loads((SAMPLES / f"{name}.json").read_text()))


def test_invalid_sample_is_rejected_by_the_schema():
    with pytest.raises(ValidationError):
        Patient(**json.loads((SAMPLES / "invalid.json").read_text()))
