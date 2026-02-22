from src.data_loader import load_all_subsets_and_aggregate, SUBSETS
from src.baseline_rag import make_embedder, build_baseline_index, retrieve_topk, build_prompt_from_chunks

def main():
    aggregated = load_all_subsets_and_aggregate(subsets=SUBSETS, k=10, split="test")

    embedder = make_embedder("st", model_name="BAAI/bge-m3")

    subset = "musique"
    agg = aggregated[subset]

    print(f"Building baseline index for {subset} ...")
    index = build_baseline_index(agg, embedder=embedder, chunk_size=512, overlap=50)

    query = agg.questions[0]
    print("\nQuery:", query)

    results = retrieve_topk(index, embedder=embedder, query=query, top_k=10)

    print("\nTop-3 retrieved chunks:")
    for r in results[:3]:
        preview = r.chunk.text[:120].replace("\n", " ")
        print(f"- score={r.score:.4f} doc={r.chunk.doc_id} chunk={r.chunk.chunk_id} text_preview={preview}...")

    prompt = build_prompt_from_chunks(query, results)
    print("\nPrompt preview (first 800 chars):")
    print(prompt[:800])

    print("\nOK baseline retrieval ran.")

if __name__ == "__main__":
    main()
