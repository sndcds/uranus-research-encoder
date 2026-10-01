"""Offline CPU inference using the reviewed pinned upstream Jina ONNX graph."""

import hashlib
import json
import threading
from pathlib import Path

from .contracts import EmbeddingKind
from .model import cached_snapshot, offline_environment, runtime_name
from .model_artifacts import ONNX_SHA256
from .version import DIMENSIONS


class OnnxBackend:
    backend = "onnx"
    max_tokens = 8192

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
        offline_environment()
        import onnxruntime as ort
        from transformers import AutoTokenizer

        ort.disable_telemetry_events()
        snapshot = cached_snapshot(self.root, "onnx")
        # Hashes bind task banks, external-data paths, and graph structure to the
        # reviewed upstream artifact. No caller can supply a model filename.
        for name, expected in ONNX_SHA256.items():
            with (snapshot / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("onnx_artifact_mismatch")
        config = json.loads((snapshot / "config.json").read_text())
        if (
            config.get("model_type") != "jina_embeddings_v3"
            or config.get("hidden_size") != DIMENSIONS
        ):
            raise ValueError("model_mismatch")
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.intra_op_threads
        options.inter_op_num_threads = self.inter_op_threads
        # Sequential graph execution bounds concurrent intermediates. Inter-op
        # threads are explicit but inactive in this mode; intra-op is tunable.
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The upstream graph materializes vocabulary-sized LoRA updates. Do not
        # retain their large temporary buffers in a process-lifetime arena.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.log_severity_level = 4
        session = ort.InferenceSession(
            str(snapshot / "onnx/model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        session.disable_fallback()
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("unexpected_execution_provider")
        inputs = {item.name: item for item in session.get_inputs()}
        if set(inputs) != {"input_ids", "attention_mask", "task_id"}:
            raise ValueError("onnx_input_mismatch")
        if any(item.type != "tensor(int64)" for item in inputs.values()):
            raise ValueError("onnx_input_mismatch")
        if inputs["task_id"].shape != [] or any(
            len(inputs[name].shape) != 2 for name in ("input_ids", "attention_mask")
        ):
            raise ValueError("onnx_input_mismatch")
        outputs = {item.name: item for item in session.get_outputs()}
        output = outputs.get("text_embeds")
        if (
            output is None
            or output.type != "tensor(float)"
            or len(output.shape) != 3
            or output.shape[-1] != DIMENSIONS
        ):
            raise ValueError("onnx_output_mismatch")
        self._session = session
        self._tokenizer = tokenizer
        self.revision = snapshot.name
        self.tokenizer_available = True
        self.loaded = True

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def embed(self, texts: list[str], kind: EmbeddingKind) -> list[list[float]]:
        import numpy as np

        # Bank mapping is verified against all native query/passage adapter
        # tensors; see ONNX_VALIDATION.md. Task selection is scalar per batch.
        task = {"query": 0, "passage": 1}[kind]
        vectors = []
        # Match the reference's one-text forward passes: request batching must
        # not change padding, numerical results, or peak activation memory.
        with self._lock:
            for text in texts:
                encoded = self._tokenizer(
                    text, add_special_tokens=True, truncation=False, return_tensors="np"
                )
                if encoded["input_ids"].shape[1] > self.max_tokens:
                    raise ValueError("token_limit")
                feed = {
                    name: np.asarray(encoded[name], dtype=np.int64)
                    for name in ("input_ids", "attention_mask")
                }
                feed["task_id"] = np.asarray(task, dtype=np.int64)
                (hidden,) = self._session.run(["text_embeds"], feed)
                if hidden.dtype != np.float32 or hidden.shape != (
                    1,
                    feed["input_ids"].shape[1],
                    DIMENSIONS,
                ):
                    raise RuntimeError("invalid_token_embeddings")
                mask = feed["attention_mask"].astype(np.float32)[..., None]
                pooled = (hidden * mask).sum(axis=1, dtype=np.float32) / np.maximum(
                    mask.sum(axis=1, dtype=np.float32), np.float32(1)
                )
                norm = np.sqrt(np.sum(pooled * pooled, axis=1, keepdims=True, dtype=np.float32))
                vector = pooled / np.maximum(norm, np.float32(1e-12))
                if (
                    vector.shape != (1, DIMENSIONS)
                    or not np.isfinite(vector).all()
                    or not np.allclose(np.linalg.norm(vector, axis=1), 1, rtol=0, atol=1e-6)
                ):
                    raise RuntimeError("invalid_vector")
                vectors.append(vector[0].tolist())
        return vectors
