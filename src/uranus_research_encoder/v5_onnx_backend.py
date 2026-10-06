"""Explicit, integrity-checked FP32 v5 backend. No Torch/PEFT inference dependency."""

import threading
from importlib.metadata import version
from pathlib import Path

from .contracts import EmbeddingKind
from .model import TorchBackend, offline_environment
from .v5_onnx_artifacts import BACKEND, verify_manifest
from .version import DIMENSIONS, MODEL_REVISION


def runtime_name() -> str:
    return f"onnxruntime-{version('onnxruntime')}-cpu"


class V5MergedOnnxBackend:
    backend = BACKEND
    max_tokens = 32768
    _input = staticmethod(TorchBackend._input)

    def __init__(
        self,
        root: Path,
        *,
        manifest_sha256: str,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
    ):
        if not 1 <= intra_op_threads <= 8 or inter_op_threads != 1:
            raise ValueError("invalid_thread_count")
        self.root = root
        self.manifest_sha256 = manifest_sha256
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        self._lock = threading.Lock()
        self.loaded = self.tokenizer_available = False
        self.revision = ""
        self._session = self._tokenizer = None

    @property
    def runtime(self) -> str:
        return runtime_name()

    def load(self) -> None:
        self.loaded = self.tokenizer_available = False
        self.revision = ""
        self._session = self._tokenizer = None
        offline_environment()
        manifest = verify_manifest(self.root, self.manifest_sha256)
        for package in ("onnxruntime", "transformers"):
            if version(package) != manifest["versions"][package]:
                raise ValueError("onnx_runtime_version_mismatch")
        # Use the same native tokenizer class/configuration as TorchBackend.
        # Loading Qwen3Config does not instantiate a model or require Torch/PEFT.
        import json

        import onnxruntime as ort
        from transformers import AutoTokenizer, Qwen3Config

        raw = json.loads((self.root / "config.json").read_text())
        for key in ("auto_map", "model_type", "architectures", "task_names"):
            raw.pop(key, None)
        tokenizer = AutoTokenizer.from_pretrained(
            self.root, config=Qwen3Config(**raw), local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "right"
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.intra_op_threads
        options.inter_op_num_threads = self.inter_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(self.root / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (
            [(v.name, v.type, v.shape) for v in inputs]
            != [
                ("input_ids", "tensor(int64)", ["batch", "sequence"]),
                ("attention_mask", "tensor(int64)", ["batch", "sequence"]),
            ]
            or [(v.name, v.type, v.shape) for v in outputs]
            != [("text_embeds", "tensor(float)", ["batch", DIMENSIONS])]
            or session.get_providers() != ["CPUExecutionProvider"]
        ):
            raise ValueError("onnx_graph_contract_mismatch")
        self._session, self._tokenizer = session, tokenizer
        self.revision = MODEL_REVISION
        self.loaded = self.tokenizer_available = True

    def count(self, text: str, kind: EmbeddingKind = "passage") -> int:
        return TorchBackend.count(self, text, kind)

    def embed(self, texts: list[str], kind: EmbeddingKind) -> list[list[float]]:
        import numpy as np

        if not self.loaded:
            raise RuntimeError("model_not_loaded")
        vectors = []
        # Preserve the native service's bounded-memory one-text-per-forward
        # behavior. Padded graph batches are validated and benchmarked separately.
        with self._lock:
            for text in texts:
                encoded = self._tokenizer(
                    self._input(text, kind),
                    add_special_tokens=True,
                    truncation=False,
                    return_tensors="np",
                )
                if encoded["input_ids"].shape[1] > self.max_tokens:
                    raise ValueError("token_limit")
                vector = self._session.run(
                    ["text_embeds"],
                    {
                        k: np.asarray(encoded[k], dtype=np.int64)
                        for k in ("input_ids", "attention_mask")
                    },
                )[0]
                if (
                    vector.dtype != np.float32
                    or vector.shape != (1, DIMENSIONS)
                    or not np.isfinite(vector).all()
                    or not np.all(
                        np.abs(np.linalg.norm(vector.astype(np.float64), axis=1) - 1) <= 1e-5
                    )
                ):
                    raise RuntimeError("invalid_vector")
                # Pooling and normalization already happened exactly once in ONNX.
                vectors.append(vector[0].tolist())
        return vectors
