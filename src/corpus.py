from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import re

from src.data_loader import AggregatedCorpus


@dataclass(frozen=True)
class Chunk:
    subset: str
    doc_id: str
    chunk_id: str
    text: str


def _simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer (good enough for a baseline index builder).
    If you later switch to a true tokenizer (tiktoken / transformers), keep the interface.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []


def _merge_with_overlap(chunks: List[str], overlap_chars: int) -> List[str]:
    if overlap_chars <= 0 or len(chunks) <= 1:
        return [c.strip() for c in chunks if c.strip()]

    out: List[str] = []
    for i, chunk in enumerate(chunks):
        current = chunk.strip()
        if not current:
            continue
        if i == 0:
            out.append(current)
            continue
        prev_tail = out[-1][-overlap_chars:]
        out.append((prev_tail + current).strip())
    return out


def chunk_document_recursive_chars(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Recursive character splitting (guide-aligned baseline splitter).
    We approximate token constraints by converting token budget to character budget
    (about 4 chars/token), then apply recursive separators.
    """
    assert chunk_size > 0
    assert 0 <= overlap < chunk_size

    max_chars = chunk_size * 4
    overlap_chars = overlap * 4

    separators = ["\n\n", "\n", ". ", " "]

    def _split_recursive(s: str, sep_idx: int = 0) -> List[str]:
        s = s.strip()
        if not s:
            return []
        if len(s) <= max_chars:
            return [s]
        if sep_idx >= len(separators):
            # hard fallback
            parts = [s[i : i + max_chars] for i in range(0, len(s), max_chars)]
            return [p.strip() for p in parts if p.strip()]

        sep = separators[sep_idx]
        raw = s.split(sep)
        if len(raw) == 1:
            return _split_recursive(s, sep_idx + 1)

        built: List[str] = []
        cur = ""
        for piece in raw:
            piece = piece.strip()
            if not piece:
                continue
            candidate = f"{cur}{sep if cur else ''}{piece}".strip()
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                if cur:
                    built.extend(_split_recursive(cur, sep_idx + 1))
                cur = piece
        if cur:
            built.extend(_split_recursive(cur, sep_idx + 1))
        return built

    return _merge_with_overlap(_split_recursive(text), overlap_chars=overlap_chars)


def build_unified_chunk_index(
    agg: AggregatedCorpus,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Builds the SINGLE unified chunk pool ("unified index") for one subset:
    - Input: 10 docs (agg.documents)
    - Output: list of Chunk objects across all docs
    """
    out: List[Chunk] = []

    for doc_idx, (doc_id, doc_text) in enumerate(zip(agg.doc_ids, agg.documents)):
        doc_chunks = chunk_document_recursive_chars(doc_text, chunk_size=chunk_size, overlap=overlap)

        for j, ch in enumerate(doc_chunks):
            out.append(
                Chunk(
                    subset=agg.subset,
                    doc_id=doc_id,
                    chunk_id=f"{agg.subset}-doc{doc_idx}-chunk{j}",
                    text=ch,
                )
            )

    return out


def summarize_index(chunks: List[Chunk]) -> Dict[str, Any]:
    """
    Small helper to sanity-check the index.
    """
    by_doc: Dict[str, int] = {}
    total_tokens = 0

    for c in chunks:
        by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
        total_tokens += len(_simple_tokenize(c.text))

    return {
        "num_chunks": len(chunks),
        "chunks_per_doc": by_doc,
        "approx_total_tokens": total_tokens,
        "avg_chunk_tokens": (total_tokens / max(len(chunks), 1)),
    }
