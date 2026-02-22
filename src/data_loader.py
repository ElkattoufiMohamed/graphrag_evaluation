from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datasets import load_dataset


# LongBench exposes WikiMQA under the HF config name "2wikimqa".
SUBSETS = ["musique", "2wikimqa", "narrativeqa", "qasper"]



@dataclass(frozen=True)
class Sample:
    """One LongBench sample (normalized)."""
    subset: str
    sample_id: str
    question: str          # LongBench uses "input"
    context: str           # LongBench uses "context"
    answers: List[str]     # LongBench uses "answers"


@dataclass(frozen=True)
class AggregatedCorpus:
    """Crucial artifact: 10 contexts aggregated into ONE unified pool/index."""
    subset: str
    documents: List[str]           # list of 10 contexts (each is a "document")
    doc_ids: List[str]             # stable IDs aligned with documents
    questions: List[str]           # 10 questions (one per sample)
    answers_list: List[List[str]]  # 10 ground-truth answer lists


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _normalize_answers(ans: Any) -> List[str]:
    if ans is None:
        return []
    if isinstance(ans, list):
        return [str(a) for a in ans]
    return [str(ans)]


def load_longbench_topk(
    subset: str,
    k: int = 10,
    split: str = "test",
    trust_remote_code: bool = True,
) -> List[Sample]:
    """
    Load LongBench subset and return top-k samples as normalized Sample objects.
    LongBench fields: input (question), context, answers, _id.
    """
    ds = load_dataset("THUDM/LongBench", subset, split=split, trust_remote_code=trust_remote_code)

    samples: List[Sample] = []
    # "Top 10" per instructions: we take first k in the split ordering
    for i in range(min(k, len(ds))):
        row = ds[i]
        sample_id = _safe_str(row.get("_id", f"{subset}-{i}"))
        question = _safe_str(row.get("input", ""))   # LongBench uses "input"
        context = _safe_str(row.get("context", ""))  # LongBench uses "context"
        answers = _normalize_answers(row.get("answers", []))

        samples.append(Sample(
            subset=subset,
            sample_id=sample_id,
            question=question,
            context=context,
            answers=answers,
        ))

    return samples


def build_aggregated_corpus(samples: List[Sample], subset: Optional[str] = None) -> AggregatedCorpus:
    """
    Crucial step: extract the 10 contexts and aggregate into a single unified 'index input'
    (represented here as a list of 10 documents + doc_ids).

    Retrieval (baseline or GraphRAG) must answer each query by selecting the correct info
    from this mixed pool of 10 documents.
    """
    if not samples:
        raise ValueError("samples is empty")

    inferred_subset = subset or samples[0].subset
    if any(s.subset != inferred_subset for s in samples):
        raise ValueError("All samples must come from the same subset for per-subset aggregation.")

    documents = [s.context for s in samples]
    doc_ids = [s.sample_id for s in samples]
    questions = [s.question for s in samples]
    answers_list = [s.answers for s in samples]

    return AggregatedCorpus(
        subset=inferred_subset,
        documents=documents,
        doc_ids=doc_ids,
        questions=questions,
        answers_list=answers_list,
    )


def load_all_subsets_and_aggregate(
    subsets: List[str] = SUBSETS,
    k: int = 10,
    split: str = "test",
) -> Dict[str, AggregatedCorpus]:
    """
    For each subset:
      - load top-k samples
      - aggregate their contexts into a SINGLE unified pool/index (10 docs)
    Returns dict: subset -> AggregatedCorpus
    """
    out: Dict[str, AggregatedCorpus] = {}
    for subset in subsets:
        samples = load_longbench_topk(subset=subset, k=k, split=split, trust_remote_code=True)
        agg = build_aggregated_corpus(samples=samples, subset=subset)
        out[subset] = agg
    return out


if __name__ == "__main__":
    aggregated = load_all_subsets_and_aggregate()

    for subset, agg in aggregated.items():
        print(f"\n=== {subset} ===")
        print(f"Docs in unified pool: {len(agg.documents)}")
        print(f"Doc ID[0]: {agg.doc_ids[0]}")
        print(f"Question[0] (first 200 chars): {agg.questions[0][:200]}")
        print(f"Context[0] (first 300 chars): {agg.documents[0][:300]}")
        print(f"Answers[0]: {agg.answers_list[0]}")
