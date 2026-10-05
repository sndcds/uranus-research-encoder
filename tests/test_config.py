import pytest
from conftest import KEY
from pydantic import ValidationError

from uranus_research_encoder.config import Settings


@pytest.mark.parametrize("key", ["short", "x" * 31, "x" * 32 + "\n", "é" * 32, "x" * 4097])
def test_invalid_key(key):
    with pytest.raises(ValidationError):
        Settings(api_key=key, jina_noncommercial=True)


def test_ack_required():
    with pytest.raises(ValidationError):
        Settings(api_key=KEY)


def test_file_preferred_and_no_dotenv(tmp_path, monkeypatch):
    secret = tmp_path / "secret"
    secret.write_text(KEY + "\n")
    (tmp_path / ".env").write_text("ENCODER_MODEL=unwanted\nJINA_NONCOMMERCIAL=0")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENCODER_API_KEY_FILE", str(secret))
    monkeypatch.setenv("ENCODER_API_KEY", "bad")
    monkeypatch.setenv("JINA_NONCOMMERCIAL", "1")
    settings = Settings.from_env()
    assert settings.api_key.get_secret_value() == KEY
    assert KEY not in repr(settings)
    assert settings.model == "jina-v5"


@pytest.mark.parametrize(
    "change",
    [
        {"model_root": "relative"},
        {"model": "other"},
        {"model_revision": "main"},
        {"max_concurrent_requests": 0},
        {"max_concurrent_requests": 9},
        {"backend": "cuda"},
        {"onnx_intra_op_threads": 0},
        {"onnx_intra_op_threads": 9},
        {"onnx_inter_op_threads": 0},
        {"backend": "onnx-merged"},
        {"backend": "onnx-merged", "merged_onnx_root": "relative"},
        {"backend": "onnx-merged", "merged_onnx_root": "/derived", "onnx_inter_op_threads": 2},
        {"merged_onnx_optimization": "extended"},
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValidationError):
        Settings(**({"api_key": KEY, "jina_noncommercial": True} | change))


def test_backend_environment_is_explicit(monkeypatch):
    monkeypatch.delenv("ENCODER_API_KEY_FILE", raising=False)
    monkeypatch.setenv("ENCODER_API_KEY", KEY)
    monkeypatch.setenv("JINA_NONCOMMERCIAL", "1")
    monkeypatch.delenv("ENCODER_BACKEND", raising=False)
    assert Settings.from_env().backend == "torch"
    monkeypatch.setenv("ENCODER_BACKEND", "onnx")
    monkeypatch.setenv("ENCODER_ONNX_INTRA_OP_THREADS", "2")
    with pytest.raises(ValidationError, match="onnx_not_supported_for_jina_v5"):
        Settings.from_env()
    monkeypatch.setenv("ENCODER_BACKEND", "onnx-merged")
    monkeypatch.setenv("ENCODER_MERGED_ONNX_ROOT", "/derived")
    with pytest.raises(ValidationError, match="onnx_not_supported_for_jina_v5"):
        Settings.from_env()
