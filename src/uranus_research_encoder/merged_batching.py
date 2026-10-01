"""Benchmark-only padded batching; the HTTP backend keeps sequential inference."""

from .version import DIMENSIONS


def embed_padded(backend, texts, kind, *, batch_size=4):
    """Run bounded, right-padded groups through one fixed task graph.

    Kept separate from embed() so batching has its own parity and timing gate.
    """
    import numpy as np

    if not 1 <= batch_size <= 16:
        raise ValueError("invalid_batch_size")
    if backend._tokenizer.padding_side != "right":
        raise ValueError("right_padding_required")
    vectors = []
    with backend._lock:
        session = backend._sessions[kind]
        for start in range(0, len(texts), batch_size):
            group = texts[start : start + batch_size]
            encoded = backend._tokenizer(
                group, padding=True, add_special_tokens=True, truncation=False, return_tensors="np"
            )
            feed = {
                name: np.asarray(encoded[name], dtype=np.int64)
                for name in ("input_ids", "attention_mask")
            }
            if feed["input_ids"].shape[1] > backend.max_tokens:
                raise ValueError("token_limit")
            (hidden,) = session.run(["text_embeds"], feed)
            if hidden.dtype != np.float32 or hidden.shape != (
                len(group),
                feed["input_ids"].shape[1],
                DIMENSIONS,
            ):
                raise RuntimeError("invalid_token_embeddings")
            mask = feed["attention_mask"].astype(np.float32)[..., None]
            pooled = (hidden * mask).sum(axis=1, dtype=np.float32) / np.maximum(
                mask.sum(axis=1, dtype=np.float32), np.float32(1)
            )
            norm = np.sqrt(np.sum(pooled * pooled, axis=1, keepdims=True, dtype=np.float32))
            normalized = pooled / np.maximum(norm, np.float32(1e-12))
            if not np.isfinite(normalized).all() or not np.allclose(
                np.linalg.norm(normalized, axis=1), 1, rtol=0, atol=1e-6
            ):
                raise RuntimeError("invalid_vector")
            vectors.extend(normalized.tolist())
    return vectors
