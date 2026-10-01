import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "context",
    [
        None,
        {"scope": "event"},
        {"scope": "venue", "venue_id": "00000000-0000-0000-0000-000000000002"},
        {
            "scope": "space",
            "venue_id": "00000000-0000-0000-0000-000000000002",
            "space_id": "00000000-0000-0000-0000-000000000003",
        },
        {
            "scope": "occurrence",
            "venue_id": "00000000-0000-0000-0000-000000000002",
            "space_id": "00000000-0000-0000-0000-000000000003",
            "occurrence_id": "00000000-0000-0000-0000-000000000004",
        },
    ],
)
def test_section_regressions(client, auth, context):
    section = {"kind": "content", "text": "Titel: Communitytreffen", "context": context}
    response = client.post(
        "/chunks",
        headers=auth,
        json={
            "model": "jina-v3",
            "documents": [
                {
                    "entity_id": "019d5f3a-de7e-780a-9fa0-dc24e5545e2e",
                    "sections": [section],
                }
            ],
        },
    )
    assert response.status_code == 200
    contexts = response.json()["documents"][0]["chunks"][0]["contexts"]
    assert len(contexts) == (0 if context is None else 1)
    if context:
        assert all(contexts[0][key] == value for key, value in context.items())


def test_admin_fixture(client, auth):
    fixture = Path(__file__).parent / "fixtures/uranus_admin_chunks_v1.json"
    data = json.loads(fixture.read_text())
    response = client.post("/chunks", headers=auth, json=data)
    assert response.status_code == 200
    examples = json.loads((fixture.parent / "http_examples_fake_backend.json").read_text())
    assert response.json() == examples["/chunks"]
    result = response.json()["documents"][0]
    assert result["entity_id"] == data["documents"][0]["entity_id"]
    assert [c["chunk_kind"] for c in result["chunks"]] == ["content", "tickets", "location_context"]
    assert result["chunks"][2]["contexts"][0]["venue_id"] == "019d9b4e-b7ca-7b1f-8d19-56ca89e33ce7"
