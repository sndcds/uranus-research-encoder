"""Compatibility import only: ONNX has not been validated for Jina v5."""

import threading
from pathlib import Path

from .contracts import EmbeddingKind
from .model import runtime_name


class OnnxBackend:
    backend = "onnx"
    max_tokens = 0

    def __init__(self, root: Path, *, intra_op_threads: int = 1, inter_op_threads: int = 1):
        if not 1 <= intra_op_threads <= 8 or not 1 <= inter_op_threads <= 8:
            raise ValueError("invalid_thread_count")
        self.root = root
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        self.loaded = False
        self.tokenizer_available = False
        self.revision = ""
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None

    @property
    def runtime(self) -> str:
        return runtime_name("onnx")

    def load(self) -> None:
        raise ValueError("onnx_not_supported_for_jina_v5")

    def count(self, text: str, kind: EmbeddingKind = "passage") -> int:
        raise ValueError("onnx_not_supported_for_jina_v5")

    def embed(self, texts: list[str], kind: EmbeddingKind) -> list[list[float]]:
        raise ValueError("onnx_not_supported_for_jina_v5")
