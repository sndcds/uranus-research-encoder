from itertools import product
from uuid import UUID

import pytest
from pydantic import ValidationError

from uranus_research_encoder.contracts import EvidenceContext

ID = UUID(int=2)


@pytest.mark.parametrize("scope", ["event", "venue", "space", "occurrence"])
@pytest.mark.parametrize("venue,space,occurrence", list(product([None, ID], repeat=3)))
def test_all_scope_combinations(scope, venue, space, occurrence):
    valid = {
        "event": venue is space is occurrence is None,
        "venue": venue is not None and space is occurrence is None,
        "space": space is not None and occurrence is None,
        "occurrence": occurrence is not None,
    }[scope]
    data = dict(scope=scope, venue_id=venue, space_id=space, occurrence_id=occurrence)
    if valid:
        assert EvidenceContext(**data).model_dump() == data
    else:
        with pytest.raises(ValidationError):
            EvidenceContext(**data)


def test_context_extra_and_unknown_scope():
    for data in ({"scope": "other"}, {"scope": "event", "unknown": 1}, {"venue_id": ID}):
        with pytest.raises(ValidationError):
            EvidenceContext(**data)
