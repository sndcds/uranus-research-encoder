"""Compatibility import only: ONNX has not been validated for Jina v5."""

import threading
from pathlib import Path

from .contracts import EmbeddingKind
from .model import runtime_name


class MergedOnnxBackend:
    backend = "onnx-merged"
    max_tokens = 0

    def __init__(
        self,
        root: Path,
        *,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        optimization: str = "disabled",
    ):
        if not 1 <= intra_op_threads <= 8 or inter_op_threads != 1:
            raise ValueError("invalid_thread_count")
        if optimization not in ("disabled", "basic"):
            raise ValueError("invalid_graph_optimization")
        self.root = root
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        self.optimization = optimization
        self.loaded = False
        self.tokenizer_available = False
        self.revision = ""
        self._lock = threading.Lock()
        self._sessions = {}
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
