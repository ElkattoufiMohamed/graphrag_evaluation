from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.metrics import best_of_many


def _first_jsonl(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            return json.loads(line)
    return None


def _mean_token(path: Path, key: str) -> float:
    if not path.exists():
        return 0.0
    vals = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if key in obj:
                vals.append(float(obj[key]))
    return sum(vals) / len(vals) if vals else 0.0


def _analysis_lines(base_path: Path, graph_path: Path) -> tuple[str, str]:
    if not base_path.exists() or not graph_path.exists():
        return (
            "Insufficient data: run baseline and GraphRAG predictions first.",
            "Insufficient data: run baseline and GraphRAG predictions first.",
        )

    base_scores = {}
    with base_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            f1, _ = best_of_many(obj.get("pred_baseline", ""), obj.get("answers", []))
            base_scores[(obj.get("subset"), obj.get("sample_id"))] = f1

    wins = 0
    losses = 0
    retrieval_modes = {"local": 0, "global": 0}

    with graph_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            key = (obj.get("subset"), obj.get("sample_id"))
            if key not in base_scores:
                continue
            f1, _ = best_of_many(obj.get("pred_graphrag", ""), obj.get("answers", []))
            if f1 > base_scores[key]:
                wins += 1
            elif f1 < base_scores[key]:
                losses += 1

            mode = obj.get("retrieval_mode_used", "local")
            if mode in retrieval_modes:
                retrieval_modes[mode] += 1

    cross_doc = (
        f"GraphRAG outperformed baseline on {wins} samples and underperformed on {losses} samples "
        f"(F1-based pairwise comparison across matched subset/sample IDs). "
        f"Retrieval mode usage: local={retrieval_modes['local']}, global={retrieval_modes['global']}."
    )
    error = (
        "Error focus should target samples where baseline wins: inspect whether entity/relation extraction or community "
        "summaries omitted key facts, and whether local/global mode selection mismatched the question's reasoning depth."
    )
    return cross_doc, error


def main(
    comparison_csv: str = "outputs/performance_comparison.csv",
    baseline_jsonl: str = "outputs/baseline_predictions.jsonl",
    graphrag_jsonl: str = "outputs/graphrag_predictions.jsonl",
    out_md: str = "outputs/final_report.md",
):
    cmp_path = Path(comparison_csv)
    base_path = Path(baseline_jsonl)
    graph_path = Path(graphrag_jsonl)

    if not cmp_path.exists():
        print("Missing performance_comparison.csv. Run src/eval_compare.py first.")
        return

    base_first = _first_jsonl(base_path)
    graph_first = _first_jsonl(graph_path)

    llm_model = "N/A"
    embedding_model = "N/A"
    if base_first and "run_meta" in base_first:
        llm_model = base_first["run_meta"].get("llm_model", llm_model)
        embedding_model = base_first["run_meta"].get("embedding_model", embedding_model)
    elif graph_first and "config" in graph_first:
        llm_model = graph_first["config"].get("llm_model", llm_model)
        embedding_model = graph_first["config"].get("embedding_model", embedding_model)

    vector_tokens = round(_mean_token(base_path, "vector_index_tokens_est"), 2)
    graph_tokens = round(_mean_token(graph_path, "graph_construction_tokens_est"), 2)

    df = pd.read_csv(cmp_path)

    lines = []
    lines.append("# GraphRAG vs Standard Vector RAG Evaluation Report")
    lines.append("")
    lines.append("## 1) Configuration Table")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| LLM Model | {llm_model} |")
    lines.append(f"| Embedding Model | {embedding_model} |")
    lines.append(f"| Avg Token Consumption (Graph Construction) | {graph_tokens} |")
    lines.append(f"| Avg Token Consumption (Vector Indexing) | {vector_tokens} |")
    lines.append("")
    lines.append("## 2) Performance Comparison Matrix")
    lines.append("")
    lines.append("| DATASET | METRIC | STANDARD BASELINE (MEAN) | GRAPHRAG (MEAN) | IMPROVEMENT (Δ) |")
    lines.append("|---|---|---:|---:|---:|")

    for _, r in df.iterrows():
        delta = float(r["improvement_delta_percent"])
        delta_txt = f"{delta:+.2f}%"
        lines.append(
            f"| {r['dataset']} | {r['metric']} | {float(r['standard_baseline_mean']):.4f} | {float(r['graphrag_mean']):.4f} | {delta_txt} |"
        )

    lines.append("")
    cross_doc, error = _analysis_lines(base_path, graph_path)

    lines.append("## 3) Critical Analysis")
    lines.append("")
    lines.append(f"- Cross-Document Reasoning: {cross_doc}")
    lines.append(f"- Error Analysis: {error}")
    lines.append("")
    lines.append("## 4) Artifacts")
    lines.append("")
    lines.append("- `src/run_baseline.py`")
    lines.append("- `src/run_graphrag.py`")
    lines.append("- `src/eval_compare.py`")

    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
