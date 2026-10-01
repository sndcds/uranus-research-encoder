"""One offline native Jina model with serialized retrieval adapter selection."""

import os
import threading
from pathlib import Path
from typing import Protocol

from .contracts import EmbeddingKind
from .version import DIMENSIONS, MODEL_REPOSITORY, MODEL_REVISION


class Backend(Protocol):
    loaded: bool
    tokenizer_available: bool
    revision: str
    max_tokens: int

    def load(self) -> None: ...
    def count(self, text: str) -> int: ...
    def embed(self, texts: list[str], kind: EmbeddingKind) -> list[list[float]]: ...


class JinaBackend:
    max_tokens = 8192

    def __init__(self, root: Path):
        self.root = root
        self.loaded = False
        self.tokenizer_available = False
        self.revision = ""
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        # Set before importing HF libraries; all loads also explicitly enforce offline.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        snapshot = Path(
            snapshot_download(
                MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                cache_dir=str(self.root),
                local_files_only=True,
            )
        )
        if snapshot.name != MODEL_REVISION:
            raise ValueError("revision_mismatch")
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        model, loading = AutoModel.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("incomplete_weights")
        if (
            model.config.model_type != "jina_embeddings_v3"
            or model.config.hidden_size != DIMENSIONS
        ):
            raise ValueError("model_mismatch")
        for task in ("retrieval_query", "retrieval_passage"):
            if not (snapshot / task / "adapter_model.safetensors").is_file():
                raise ValueError("missing_adapter")
            adapter_loading = model.load_adapter(
                str(snapshot / task),
                adapter_name=task,
                adapter_kwargs={"local_files_only": True},
            )
            # Transformers reports previously loaded adapters as absent from the
            # next adapter file. Only this task's keys must be present here.
            missing = [key for key in adapter_loading.missing_keys if f".{task}." in key]
            if missing or adapter_loading.mismatched_keys:
                raise ValueError("incomplete_adapter")
        model.to("cpu")
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self.revision = snapshot.name
        self.tokenizer_available = True
        self.loaded = True

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def embed(self, texts: list[str], kind: EmbeddingKind) -> list[list[float]]:
        import torch
        from torch.nn import functional as F

        vectors = []
        # Lock includes adapter mutation. One text per forward pass keeps output
        # independent of request batching/padding and bounds activation memory.
        with self._lock, torch.inference_mode():
            self._model.set_adapter("retrieval_query" if kind == "query" else "retrieval_passage")
            for text in texts:
                encoded = self._tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                    return_tensors="pt",
                )
                if encoded["input_ids"].shape[1] > self.max_tokens:
                    raise ValueError("token_limit")
                output = self._model(**encoded).last_hidden_state.float()
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vector = F.normalize(pooled, p=2, dim=1).to(torch.float32)
                if vector.shape != (1, DIMENSIONS) or not torch.isfinite(vector).all():
                    raise RuntimeError("invalid_vector")
                vectors.append(vector[0].tolist())
        return vectors
