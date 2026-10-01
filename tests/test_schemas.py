from pathlib import Path

from uranus_research_encoder.schemas import schemas


def test_committed_schemas_match():
    root = Path(__file__).resolve().parents[1] / "contracts"
    expected = schemas()
    assert {p.stem for p in root.glob("*.json")} == set(expected)
    for name, content in expected.items():
        assert (root / f"{name}.json").read_text() == content
