"""One offline native Jina model with serialized retrieval adapter selection."""

import json
import os
import threading
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from .contracts import EmbeddingKind
from .model_artifacts import BackendName, artifact_patterns
from .version import DIMENSIONS, MODEL_REPOSITORY, MODEL_REVISION


class Backend(Protocol):
    backend: str
    runtime: str
    loaded: bool
    tokenizer_available: bool
    revision: str
    max_tokens: int

    def load(self) -> None: ...
    def count(self, text: str, kind: EmbeddingKind = "passage") -> int: ...
    def embed(self, texts: list[str], kind: EmbeddingKind) -> list[list[float]]: ...


def offline_environment() -> None:
    # Set before importing HF libraries; all loads also explicitly enforce offline.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def runtime_name(backend: str) -> str:
    if backend in ("onnx", "onnx-merged"):
        return f"onnxruntime-{version('onnxruntime')}-cpu"
    return (
        f"torch-{version('torch')}-transformers-{version('transformers')}"
        f"-peft-{version('peft')}-cpu"
    )


def cached_snapshot(root: Path, backend: BackendName) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            cache_dir=str(root),
            allow_patterns=artifact_patterns(backend),
            local_files_only=True,
        )
    )
    if snapshot.name != MODEL_REVISION:
        raise ValueError("revision_mismatch")
    return snapshot


class TorchBackend:
    backend = "torch"
    max_tokens = 32768

    def __init__(self, root: Path):
        self.root = root
        self.loaded = False
        self.tokenizer_available = False
        self.revision = ""
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    @property
    def runtime(self) -> str:
        return runtime_name("torch")

    def load(self) -> None:
        self.loaded = False
        self.tokenizer_available = False
        self.revision = ""
        self._model = self._tokenizer = None
        offline_environment()
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        snapshot = cached_snapshot(self.root, "torch")
        # The upstream configuration subclasses Qwen3Config without changing it.
        # Read data only; never import the repository's custom Python wrapper.
        raw = json.loads((snapshot / "config.json").read_text())
        if (
            raw.get("model_type") != "jina_embeddings_v5"
            or raw.get("architectures") != ["JinaEmbeddingsV5Model"]
            or raw.get("hidden_size") != DIMENSIONS
            or raw.get("max_position_embeddings") != self.max_tokens
            or "retrieval" not in raw.get("task_names", [])
        ):
            raise ValueError("model_mismatch")
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            if not (snapshot / filename).is_file():
                raise ValueError("missing_tokenizer")
        adapter = snapshot / "adapters" / "retrieval"
        if not all(
            (adapter / name).is_file()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        ):
            raise ValueError("missing_adapter")
        retrieval = json.loads((adapter / "adapter_config.json").read_text())
        if (
            retrieval.get("peft_type") != "LORA"
            or retrieval.get("task_type") != "FEATURE_EXTRACTION"
            or retrieval.get("base_model_name_or_path") != MODEL_REPOSITORY
            or set(retrieval.get("target_modules", []))
            != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
            or retrieval.get("r") != 32
            or retrieval.get("lora_alpha") != 32
            or retrieval.get("lora_dropout") != 0.1
            or retrieval.get("use_dora", False)
            or retrieval.get("use_rslora", False)
            or retrieval.get("modules_to_save") is not None
            or retrieval.get("bias", "none") != "none"
        ):
            raise ValueError("retrieval_configuration_mismatch")
        native = dict(raw)
        for key in ("auto_map", "model_type", "architectures", "task_names"):
            native.pop(key, None)
        config = Qwen3Config(**native)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer.padding_side = "right"
        model, loading = Qwen3Model.from_pretrained(
            snapshot,
            config=config,
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if any(
            loading.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError("incomplete_weights")
        adapter_loading = model.load_adapter(
            str(adapter),
            adapter_name="retrieval",
            adapter_kwargs={"local_files_only": True},
        )
        missing = [key for key in adapter_loading.missing_keys if ".retrieval." in key]
        if missing or adapter_loading.mismatched_keys or adapter_loading.unexpected_keys:
            raise ValueError("incomplete_adapter")
        model.set_adapter("retrieval")
        model.to("cpu")
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self.revision = snapshot.name
        self.tokenizer_available = True
        self.loaded = True

    @staticmethod
    def _input(text: str, kind: EmbeddingKind) -> str:
        if kind not in ("query", "passage"):
            raise ValueError("embedding_kind")
        return ("Query: " if kind == "query" else "Document: ") + text

    def count(self, text: str, kind: EmbeddingKind = "passage") -> int:
        """Count the actual model input, including its retrieval prefix."""
        return len(
            self._tokenizer.encode(
                self._input(text, kind),
                add_special_tokens=True,
                truncation=False,
            )
        )

    def embed(self, texts: list[str], kind: EmbeddingKind) -> list[list[float]]:
        import torch
        from torch.nn import functional as F

        vectors = []
        # Lock includes adapter mutation. One text per forward pass keeps output
        # independent of request batching/padding and bounds activation memory.
        with self._lock, torch.inference_mode():
            self._model.set_adapter("retrieval")
            for text in texts:
                encoded = self._tokenizer(
                    self._input(text, kind),
                    add_special_tokens=True,
                    truncation=False,
                    return_tensors="pt",
                )
                if encoded["input_ids"].shape[1] > self.max_tokens:
                    raise ValueError("token_limit")
                output = self._model(**encoded, use_cache=False).last_hidden_state.float()
                last = encoded["attention_mask"].sum(dim=1) - 1
                pooled = output[torch.arange(output.shape[0]), last]
                vector = F.normalize(pooled, p=2, dim=1).to(torch.float32)
                if (
                    vector.shape != (1, DIMENSIONS)
                    or not torch.isfinite(vector).all()
                    or not torch.allclose(vector.norm(dim=1), torch.ones(1), atol=1e-6)
                ):
                    raise RuntimeError("invalid_vector")
                vectors.append(vector[0].tolist())
        return vectors


# Preserve imports used by existing integration clients while naming the reference.
JinaBackend = TorchBackend
