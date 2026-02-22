import os
import json
from datetime import datetime

from src.data_loader import load_all_subsets_and_aggregate, SUBSETS
from src.graphrag_runner import (
    build_graphrag,
    graphrag_retrieve_context,
    format_generation_prompt,
    set_llm_for_graphrag,
    reset_graphrag_llm_usage,
    get_graphrag_llm_usage,
)
from src.llm_provider import build_unified_llm_from_env
from src.token_utils import count_many_text_tokens


def _choose_retrieval_mode(question: str) -> str:
    q = question.lower()
    cues = ["why", "how", "relation", "compare", "across", "both", "together", "cause"]
    return "global" if any(c in q for c in cues) else "local"


def main():
    out_path = "outputs/graphrag_predictions.jsonl"
    os.makedirs("outputs", exist_ok=True)

    config = {
        "timestamp": datetime.now().isoformat(),
        "group": "graphrag",
        "implementation": "nano-graphrag",
        "llm_provider": None,
        "llm_model": None,
        "embedding_model": None,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k_equiv": 10,
        "retrieval_mode": "auto",
        "context_budget_tokens_est": 5120,
    }

    llm = build_unified_llm_from_env()
    set_llm_for_graphrag(llm)
    config["llm_provider"] = llm.provider
    config["llm_model"] = llm.model

    embedding_backend = os.environ.get("EVAL_EMBEDDING_BACKEND", "st")
    config["embedding_model"] = "text-embedding-3-small" if embedding_backend == "openai" else "BAAI/bge-m3"
    print("Running GraphRAG with config:", config)
    print("Writing predictions to:", out_path)

    aggregated = load_all_subsets_and_aggregate(subsets=SUBSETS, k=10, split="test")

    with open(out_path, "w", encoding="utf-8") as f:
        for subset in SUBSETS:
            agg = aggregated[subset]
            docs = agg.documents  # 10 contexts (the mixed pool)

            workdir = f"outputs/nano_graphrag_cache/{subset}"
            os.makedirs(workdir, exist_ok=True)

            rag = build_graphrag(
                working_dir=workdir,
                embed_device=os.environ.get("EMBED_DEVICE", "cpu"),
                embedding_backend=embedding_backend,
                embedding_model=config["embedding_model"],
            )

            print(f"\n[GraphRAG] Indexing subset={subset} with {len(docs)} docs ...")
            reset_graphrag_llm_usage()
            graph_construction_tokens_est = count_many_text_tokens(docs)
            rag.insert(docs)
            graph_llm_usage = get_graphrag_llm_usage()

            gen_prompt_tokens = 0
            gen_completion_tokens = 0
            gen_total_tokens = 0

            for i, (sample_id, q, gold) in enumerate(zip(agg.doc_ids, agg.questions, agg.answers_list)):
                retrieval_mode = _choose_retrieval_mode(q) if config["retrieval_mode"] == "auto" else config["retrieval_mode"]
                ctx = graphrag_retrieve_context(rag, q, mode=retrieval_mode)
                prompt = format_generation_prompt(q, ctx)

                pred = llm.generate(prompt, temperature=0.0)
                usage = llm.last_usage
                gen_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                gen_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                gen_total_tokens += int(usage.get("total_tokens", 0) or 0)

                rec = {
                    "subset": subset,
                    "sample_id": sample_id,
                    "sample_idx": i,
                    "question": q,
                    "answers": gold,
                    "pred_graphrag": pred,
                    "retrieved_context": ctx,
                    "graph_construction_tokens_est": graph_construction_tokens_est,
                    "graph_construction_llm_prompt_tokens": graph_llm_usage.get("prompt_tokens", 0),
                    "graph_construction_llm_completion_tokens": graph_llm_usage.get("completion_tokens", 0),
                    "graph_construction_llm_total_tokens": graph_llm_usage.get("total_tokens", 0),
                    "graphrag_generation_prompt_tokens": gen_prompt_tokens,
                    "graphrag_generation_completion_tokens": gen_completion_tokens,
                    "graphrag_generation_total_tokens": gen_total_tokens,
                    "retrieval_mode_used": retrieval_mode,
                    "config": config,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                print(f"  - [{subset}] sample {i+1}/10 done")

    print("\nGraphRAG run complete.")


if __name__ == "__main__":
    main()
