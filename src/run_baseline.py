from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any

from src.data_loader import load_all_subsets_and_aggregate, SUBSETS
from src.baseline_rag import make_embedder, build_baseline_index, retrieve_topk, build_prompt_from_chunks
from src.llm_provider import build_unified_llm_from_env
from src.token_utils import count_many_text_tokens


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    # ----- Fixed experiment config -----
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K = 10

    embedding_backend = os.getenv("EVAL_EMBEDDING_BACKEND", "st")  # st | openai
    if embedding_backend == "openai":
        embedder = make_embedder("openai", model="text-embedding-3-small")
        embedding_model = "text-embedding-3-small"
    else:
        embedder = make_embedder("st", model_name="BAAI/bge-m3")
        embedding_model = "BAAI/bge-m3"

    # LLM: unified across baseline and GraphRAG.
    llm = build_unified_llm_from_env()

    # Load aggregated corpora 
    aggregated = load_all_subsets_and_aggregate(subsets=SUBSETS, k=10, split="test")

    out_dir = "outputs"
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "baseline_predictions.jsonl")

    run_meta: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "embedding_model": embedding_model,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
    }

    print("Running baseline with config:", run_meta)
    print(f"Writing predictions to: {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        # For each subset: build index once, then run 10 queries
        for subset, agg in aggregated.items():
            print(f"\n[Baseline] Building index for subset={subset} ...")
            index = build_baseline_index(
                agg,
                embedder=embedder,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
            if embedding_backend == "openai":
                index_token_budget = int(getattr(embedder, "usage_tokens", 0))
            else:
                index_token_budget = count_many_text_tokens([ch.text for ch in index.chunks])

            llm_prompt_tokens = 0
            llm_completion_tokens = 0
            llm_total_tokens = 0

            for i, (qid, question, answers) in enumerate(zip(agg.doc_ids, agg.questions, agg.answers_list)):
                retrieved = retrieve_topk(index, embedder=embedder, query=question, top_k=TOP_K)
                prompt = build_prompt_from_chunks(question, retrieved)

                pred = llm.generate(prompt, temperature=0.0)
                usage = llm.last_usage
                llm_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                llm_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                llm_total_tokens += int(usage.get("total_tokens", 0) or 0)

                record = {
                    "run_meta": run_meta,
                    "vector_index_tokens_est": index_token_budget,
                    "baseline_generation_prompt_tokens": llm_prompt_tokens,
                    "baseline_generation_completion_tokens": llm_completion_tokens,
                    "baseline_generation_total_tokens": llm_total_tokens,
                    "subset": subset,
                    "sample_id": qid,
                    "sample_idx": i,
                    "question": question,
                    "answers": answers,
                    "pred_baseline": pred,
                    "retrieved": [
                        {
                            "doc_id": r.chunk.doc_id,
                            "chunk_id": r.chunk.chunk_id,
                            "score": round(r.score, 6),
                            "text_preview": r.chunk.text[:200].replace("\n", " "),
                        }
                        for r in retrieved
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"  - [{subset}] sample {i+1}/10 done")

    print("\nBaseline run complete.")


if __name__ == "__main__":
    main()
