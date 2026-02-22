import os
import asyncio
import random
import json
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs

# -------------------------
# Chunking (token-based) to match baseline hyperparams: 512 / 50
# nano-graphrag passes token IDs, we return chunk records.
# -------------------------
def chunking_by_token_size(
    tokens_list: List[List[int]],
    doc_keys: List[str],
    tiktoken_model,
    overlap_token_size: int = 50,
    max_token_size: int = 512,
):
    results = []
    for doc_i, tokens in enumerate(tokens_list):
        chunk_token_batches = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_ids = tokens[start : start + max_token_size]
            chunk_token_batches.append(chunk_ids)
            lengths.append(min(max_token_size, len(tokens) - start))

        chunk_texts = tiktoken_model.decode_batch(chunk_token_batches)
        for j, chunk in enumerate(chunk_texts):
            results.append(
                {
                    "tokens": lengths[j],
                    "content": chunk.strip(),
                    "chunk_order_index": j,
                    "full_doc_id": doc_keys[doc_i],
                }
            )
    return results



# Embeddings: BAAI/bge-m3
@dataclass
class LocalBGEEmbedder:
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    cache_dir: Optional[str] = None

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name, device=self.device, cache_folder=self.cache_dir)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.max_len = getattr(self.model, "max_seq_length", 8192)

    def encode(self, texts: List[str]) -> np.ndarray:
        # normalize_embeddings=True => cosine similarity friendly
        return self.model.encode(texts, normalize_embeddings=True)


_LLM_SEM = asyncio.Semaphore(int(os.getenv("GRAPHRAG_LLM_CONCURRENCY", "1")))
_SYNC_LLM = None
_GRAPH_LLM_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _looks_like_json_task(prompt: str) -> bool:
    p = prompt.lower()
    return ("json" in p) and (
        ("output" in p)
        or ("return" in p)
        or ("format" in p)
        or ("response_format" in p)
    )


def _extract_json_candidate(text: str) -> str:
    t = (text or "").strip()
    # strip markdown fences
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    # keep only the largest object span if extra prose exists
    left = t.find("{")
    right = t.rfind("}")
    if left != -1 and right != -1 and right > left:
        t = t[left : right + 1]
    return t


def _extract_first_json_value(text: str) -> str:
    """Extract first decodable JSON value from noisy text."""
    t = (text or "").strip()
    if not t:
        raise ValueError("empty response")

    # try direct decode first
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(t)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    # find first object/array start and decode from there
    starts = [i for i, ch in enumerate(t) if ch in "[{"]
    for s in starts:
        try:
            obj, end = decoder.raw_decode(t[s:])
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            continue

    raise ValueError("no decodable JSON value found")


def _simple_json_repair(text: str) -> str:
    t = _extract_json_candidate(text)
    # remove trailing commas before object/array closure
    t = re.sub(r",\s*([}\]])", r"\1", t)
    # close unbalanced braces/brackets if model truncated end
    open_obj = t.count("{") - t.count("}")
    open_arr = t.count("[") - t.count("]")
    if open_arr > 0:
        t += "]" * open_arr
    if open_obj > 0:
        t += "}" * open_obj
    return t


def _ensure_valid_json_or_raise(raw: str) -> str:
    c1 = _extract_json_candidate(raw)
    if c1:
        try:
            obj = json.loads(c1)
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

    return _extract_first_json_value(raw)


def _repair_with_llm(raw: str) -> str:
    repair_prompt = (
        "You will receive malformed JSON. Return ONLY valid JSON with the same fields and values. "
        "Do not add explanations, markdown, or extra text.\n\n"
        f"MALFORMED_JSON:\n{raw}"
    )
    repaired = _SYNC_LLM.generate(repair_prompt, temperature=0.0)
    return _simple_json_repair(repaired)


def set_llm_for_graphrag(llm) -> None:
    global _SYNC_LLM
    _SYNC_LLM = llm


def reset_graphrag_llm_usage() -> None:
    _GRAPH_LLM_USAGE["prompt_tokens"] = 0
    _GRAPH_LLM_USAGE["completion_tokens"] = 0
    _GRAPH_LLM_USAGE["total_tokens"] = 0


def get_graphrag_llm_usage() -> dict:
    return dict(_GRAPH_LLM_USAGE)

async def llm_complete(prompt: str, system_prompt=None, history_messages=None, **kwargs) -> str:
    if _SYNC_LLM is None:
        raise RuntimeError("GraphRAG LLM is not configured. Call set_llm_for_graphrag() first.")

    temperature = float(kwargs.get("temperature", 0.0))
    full_prompt = prompt if system_prompt is None else f"{system_prompt}\n\n{prompt}"

    async with _LLM_SEM:
        for attempt in range(8):
            try:
                def _call():
                    return _SYNC_LLM.generate(full_prompt, temperature=temperature)

                out = await asyncio.to_thread(_call)
                if _looks_like_json_task(full_prompt) and os.getenv("GRAPHRAG_JSON_REPAIR", "1") == "1":
                    try:
                        out = _ensure_valid_json_or_raise(out)
                    except Exception:
                        # local models may return empty / concatenated / fenced JSON.
                        repaired = _repair_with_llm(out)
                        try:
                            out = _ensure_valid_json_or_raise(repaired)
                        except Exception:
                            # second-pass repair with stronger instruction
                            repaired2 = _repair_with_llm(repaired)
                            out = _ensure_valid_json_or_raise(repaired2)
                usage = getattr(_SYNC_LLM, "last_usage", {})
                _GRAPH_LLM_USAGE["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                _GRAPH_LLM_USAGE["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                _GRAPH_LLM_USAGE["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
                await asyncio.sleep(0.25 + random.uniform(0.0, 0.25))
                return out
            except Exception as e:
                msg = str(e)
                if (
                    ("429" in msg)
                    or ("Too Many" in msg)
                    or ("503" in msg)
                    or ("temporarily" in msg.lower())
                    or ("timeout" in msg.lower())
                ):
                    wait = min(60, (2 ** attempt)) + random.uniform(0.0, 1.0)
                    await asyncio.sleep(wait)
                    continue
                raise


def build_graphrag(
    working_dir: str,
    embed_device: str = "cpu",
    embedding_backend: str = "st",
    embedding_model: str = "BAAI/bge-m3",
) -> GraphRAG:
    if embedding_backend == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        @wrap_embedding_func_with_attrs(
            embedding_dim=1536,
            max_token_size=8192,
        )
        async def local_embedding(texts: List[str]) -> np.ndarray:
            def _embed() -> np.ndarray:
                resp = client.embeddings.create(model=embedding_model, input=texts)
                return np.asarray([d.embedding for d in resp.data], dtype=np.float32)

            return await asyncio.to_thread(_embed)
    else:
        embedder = LocalBGEEmbedder(model_name=embedding_model, device=embed_device, cache_dir=working_dir)

        @wrap_embedding_func_with_attrs(
            embedding_dim=embedder.dim,
            max_token_size=embedder.max_len,
        )
        async def local_embedding(texts: List[str]) -> np.ndarray:
            return await asyncio.to_thread(embedder.encode, texts)

    rag = GraphRAG(
        working_dir=working_dir,
        # enforce baseline-style chunking
        chunk_func=chunking_by_token_size,
        # embeddings
        embedding_func=local_embedding,
        # LLMs: “best” and “cheap” (can be same for now)
        best_model_func=llm_complete,
        cheap_model_func=llm_complete,
    )
    return rag


def trim_context_to_budget(context: str, max_tokens_est: int = 5120) -> str:
    """
    Hard constraint in spec: GraphRAG retrieved context ~= baseline 10 chunks.
    We do a cheap estimate using whitespace tokens.
    """
    words = context.split()
    if len(words) <= max_tokens_est:
        return context
    return " ".join(words[:max_tokens_est])


def graphrag_retrieve_context(rag: GraphRAG, question: str, mode: str = "local") -> str:
    # only_need_context=True returns retrieved reports/entities instead of final answer
    qp = QueryParam(mode=mode, only_need_context=True)
    ctx = rag.query(question, param=qp)
    if not isinstance(ctx, str):
        ctx = str(ctx)
    return trim_context_to_budget(ctx, max_tokens_est=5120)


def format_generation_prompt(question: str, context: str) -> str:
    return (
        "You are answering using the provided context.\n"
        f"Question: {question}\n\n"
        "Context:\n\n"
        f"{context}\n\n"
        "Answer (be concise, factual, and only use the context):"
    )
