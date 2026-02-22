from src.data_loader import load_all_subsets_and_aggregate, SUBSETS

def main():
    aggregated = load_all_subsets_and_aggregate(subsets=SUBSETS, k=10, split="test")

    for subset in SUBSETS:
        agg = aggregated[subset]

        # Crucial checks
        assert len(agg.documents) == 10, f"{subset}: expected 10 documents"
        assert len(agg.questions) == 10, f"{subset}: expected 10 questions"
        assert len(agg.answers_list) == 10, f"{subset}: expected 10 answers lists"
        assert len(agg.doc_ids) == 10, f"{subset}: expected 10 doc_ids"

        # Content checks
        assert all(isinstance(d, str) and d.strip() for d in agg.documents), f"{subset}: empty context found"
        assert all(isinstance(q, str) and q.strip() for q in agg.questions), f"{subset}: empty input/question found"
        assert all(isinstance(a, list) for a in agg.answers_list), f"{subset}: answers_list must be list[list[str]]"

        print(f"[OK] {subset}: 10 docs pooled + 10 Q/A")

    print("\nAll subset loaders passed.")

if __name__ == "__main__":
    main()
