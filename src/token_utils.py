from __future__ import annotations

from typing import Iterable


def count_text_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens with tiktoken when available, otherwise fallback to whitespace."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text or ""))
    except Exception:
        return len((text or "").split())


def count_many_text_tokens(texts: Iterable[str], encoding_name: str = "cl100k_base") -> int:
    return sum(count_text_tokens(t, encoding_name=encoding_name) for t in texts)
