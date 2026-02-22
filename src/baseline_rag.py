from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.corpus import Chunk
from src.data_loader import AggregatedCorpus
from src.corpus import build_unified_chunk_index

# Embedding backends

class Embedder:
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0:1, :]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.usage_tokens = 0

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.usage_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        embs = [d.embedding for d in resp.data]
        return np.asarray(embs, dtype=np.float32)


def make_embedder(kind: str, **kwargs) -> Embedder:
    kind = kind.lower().strip()
    if kind == "st":
        return SentenceTransformerEmbedder(**kwargs)
    if kind == "openai":
        return OpenAIEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder kind: {kind}. Use 'st' or 'openai'.")


# Baseline DPR / Vector RAG

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass
class BaselineIndex:
    subset: str
    chunks: List[Chunk]
    chunk_embeddings: np.ndarray 


def build_baseline_index(
    agg: AggregatedCorpus,
    embedder: Embedder,
    chunk_size: int = 512,
    overlap: int = 50,
) -> BaselineIndex:
    """
    1) Ingestion: Load aggregated corpus (10 docs)
    2) Segmentation: chunk into unified pool
    3) Vectorization: embed all chunks
    """
    chunks = build_unified_chunk_index(agg, chunk_size=chunk_size, overlap=overlap)
    texts = [c.text for c in chunks]
    chunk_embs = embedder.embed_texts(texts)  # (N, D)

    return BaselineIndex(subset=agg.subset, chunks=chunks, chunk_embeddings=chunk_embs)


def retrieve_topk(
    index: BaselineIndex,
    embedder: Embedder,
    query: str,
    top_k: int = 10,
) -> List[RetrievalResult]:
    """
    4) Retrieval: cosine similarity search to retrieve top-k chunks
    """
    q_emb = embedder.embed_query(query)  # (1, D)
    sims = cosine_similarity(q_emb, index.chunk_embeddings)[0]  # (N,)

    # Top-k indices (descending similarity)
    top_idx = np.argsort(-sims)[:top_k]
    results = [RetrievalResult(chunk=index.chunks[i], score=float(sims[i])) for i in top_idx]
    return results


def build_prompt_from_chunks(
    query: str,
    retrieved: List[RetrievalResult],
    max_chars: int = 12000,
) -> str:
    """
    5) Generation: prepare prompt context for LLM.
    Keep it simple: concatenate top chunks with separators.
    (You can later enforce token budget more strictly.)
    """
    parts = []
    parts.append("You are answering using the provided context.\n")
    parts.append(f"Question: {query}\n\n")
    parts.append("Context:\n")

    used = 0
    for r in retrieved:
        block = f"\n[DOC={r.chunk.doc_id} | CHUNK={r.chunk.chunk_id} | SCORE={r.score:.4f}]\n{r.chunk.text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)

    parts.append("\nAnswer:")
    return "".join(parts)

# Optional: LLM generation hook

def generate_with_openai_chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Simple generation function. Replace with Gemini/Qwen if needed.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer questions using the provided context. Be concise."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()
