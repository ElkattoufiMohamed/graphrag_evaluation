from src.data_loader import load_all_subsets_and_aggregate, SUBSETS
from src.corpus import build_unified_chunk_index, summarize_index

def main():
    aggregated = load_all_subsets_and_aggregate(subsets=SUBSETS, k=10, split="test")

    for subset, agg in aggregated.items():
        chunks = build_unified_chunk_index(agg, chunk_size=512, overlap=50)
        stats = summarize_index(chunks)

        print(f"\n=== {subset} ===")
        print("Docs:", len(agg.documents))
        print("Unified chunks:", stats["num_chunks"])
        print("Avg chunk tokens:", round(stats["avg_chunk_tokens"], 2))
        print("Chunks per doc (sample):", list(stats["chunks_per_doc"].items())[:3])
        print("First chunk preview:", chunks[0].text[:200].replace("\n", " ") + " ...")

        # Hard checks
        assert len(agg.documents) == 10
        assert stats["num_chunks"] > 0

    print("\nAll unified-index chunking tests passed.")

if __name__ == "__main__":
    main()
