"""Section v2 chunking using exact model-input token counts, never estimates."""

import hashlib
import re
from collections.abc import Callable

from .contracts import MAX_CHUNKS, MAX_TOKENS, OVERLAP, Chunk, Kind, Section

Count = Callable[[str], int]


def _fitting_prefix(text: str, count: Count, limit: int) -> int:
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    # Token counts need not be monotonic. The search may underfill, but a
    # verified final count makes it safe; we never trust a character estimate.
    if low == 0 or count(text[:low]) > limit:
        raise ValueError("cannot_split")
    return low


def _overlap_size(text: str, count: Count) -> int:
    low, high = 0, len(text) - 1  # Always make forward progress.
    while low < high:
        middle = (low + high + 1) // 2
        if count(text[-middle:]) <= OVERLAP:
            low = middle
        else:
            high = middle - 1
    return low if low and count(text[-low:]) <= OVERLAP else 0


def _parts(text: str, count: Count):
    remaining = text
    produced = 0
    while remaining:
        if count(remaining) <= MAX_TOKENS:
            yield remaining
            return
        cut = _fitting_prefix(remaining, count, MAX_TOKENS)
        boundaries = [
            m.end() for m in re.finditer(r"\n\n|[.!?]\s+", remaining[:cut]) if m.end() >= cut * 0.65
        ]
        if boundaries:
            candidate = remaining[: boundaries[-1]].strip()
            if candidate and count(candidate) <= MAX_TOKENS:
                cut = boundaries[-1]
        raw = remaining[:cut]
        part = raw.strip()
        if not part or not 1 <= count(part) <= MAX_TOKENS:
            raise ValueError("cannot_split")
        yield part
        produced += 1
        if produced >= MAX_CHUNKS:
            raise ValueError("chunk_limit")
        overlap = _overlap_size(raw, count)
        remaining = remaining[cut - overlap :]


def chunk_sections(sections: list[Section], count: Count) -> list[Chunk]:
    contextual = any(s.context is not None for s in sections)
    if contextual and any(s.context is None for s in sections):
        raise ValueError("mixed_context")
    if not sections:
        return []
    combined = "\n\n".join(s.text for s in sections)
    # The reviewed admin v2 contract labels a small combined plain document content.
    work = sections
    if not contextual and count(combined) <= MAX_TOKENS:
        work = [Section(kind="content", text=combined)]
    chunks: dict[tuple[Kind, str], Chunk] = {}
    for section in work:
        for text in _parts(section.text, count):
            key = (section.kind, text)
            if key not in chunks:
                if len(chunks) >= MAX_CHUNKS:
                    raise ValueError("chunk_limit")
                chunks[key] = Chunk(
                    chunk_index=len(chunks),
                    chunk_kind=section.kind,
                    text=text,
                    contexts=[],
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    token_count=count(text),
                )
            chunk = chunks[key]
            if section.context is not None and section.context not in chunk.contexts:
                chunk.contexts.append(section.context)
    for chunk in chunks.values():
        chunk.contexts.sort(key=lambda context: context.model_dump_json())
    return list(chunks.values())
