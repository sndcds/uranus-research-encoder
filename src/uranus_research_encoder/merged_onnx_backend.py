"""Experimental, locally exported task-free ONNX inference on CPU."""

import threading
from pathlib import Path

from .contracts import EmbeddingKind
from .merged_artifacts import TASKS, verify_manifest
from .model import offline_environment, runtime_name
from .version import DIMENSIONS, MODEL_REVISION


class MergedOnnxBackend:
    backend = "onnx-merged"
    max_tokens = 8192

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
        offline_environment()
        import onnxruntime as ort
        from transformers import AutoTokenizer

        manifest = verify_manifest(self.root)
        ort.disable_telemetry_events()
        tokenizer = AutoTokenizer.from_pretrained(
            self.root / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        sessions = {}
        for kind in TASKS:
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.intra_op_threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # Keep the measured baseline explicit. Basic constant folding is a
            # separate experiment with its own unchanged semantic acceptance gate.
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                if self.optimization == "basic"
                else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            options.log_severity_level = 4
            session = ort.InferenceSession(
                str(self.root / manifest["tasks"][kind]["graph"]),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            session.disable_fallback()
            if session.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("unexpected_execution_provider")
            inputs = {item.name: item for item in session.get_inputs()}
            if set(inputs) != {"input_ids", "attention_mask"} or any(
                item.type != "tensor(int64)"
                or len(item.shape) != 2
                or any(isinstance(dim, int) for dim in item.shape)
                for item in inputs.values()
            ):
                raise ValueError("onnx_input_mismatch")
            outputs = session.get_outputs()
            if len(outputs) != 1 or (
                outputs[0].name != "text_embeds"
                or outputs[0].type != "tensor(float)"
                or len(outputs[0].shape) != 3
                or outputs[0].shape[-1] != DIMENSIONS
            ):
                raise ValueError("onnx_output_mismatch")
            sessions[kind] = session
        self._sessions = sessions
        self._tokenizer = tokenizer
        self.revision = MODEL_REVISION
        self.tokenizer_available = self.loaded = True

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def embed(self, texts: list[str], kind: EmbeddingKind) -> list[list[float]]:
        import numpy as np

        vectors = []
        with self._lock:
            session = self._sessions[kind]
            # Sequential baseline. Real padding/batching is a separate experiment.
            for text in texts:
                encoded = self._tokenizer(
                    text, add_special_tokens=True, truncation=False, return_tensors="np"
                )
                feed = {
                    name: np.asarray(encoded[name], dtype=np.int64)
                    for name in ("input_ids", "attention_mask")
                }
                if feed["input_ids"].shape[1] > self.max_tokens:
                    raise ValueError("token_limit")
                (hidden,) = session.run(["text_embeds"], feed)
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
                if not np.isfinite(vector).all() or not np.allclose(
                    np.linalg.norm(vector, axis=1), 1, rtol=0, atol=1e-6
                ):
                    raise RuntimeError("invalid_vector")
                vectors.append(vector[0].tolist())
        return vectors
