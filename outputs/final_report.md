# GraphRAG vs Standard Vector RAG Evaluation Report

## 1) Configuration Table

| Item | Value |
|---|---|
| LLM Model | qcwind/qwen2.5-7B-instruct-Q4_K_M |
| Embedding Model | BAAI/bge-m3 |
| Avg Token Consumption (Graph Construction) | 131812.75 |
| Avg Token Consumption (Vector Indexing) | 148751.25 |

## 2) Performance Comparison Matrix

| DATASET | METRIC | STANDARD BASELINE (MEAN) | GRAPHRAG (MEAN) | IMPROVEMENT (Δ) |
|---|---|---:|---:|---:|
| 2wikimqa | F1-Score | 0.0350 | 0.0808 | +131.06% |
| 2wikimqa | ROUGE-L | 0.0350 | 0.0808 | +131.06% |
| musique | F1-Score | 0.0630 | 0.0598 | -5.13% |
| musique | ROUGE-L | 0.0565 | 0.0547 | -3.23% |
| narrativeqa | F1-Score | 0.0595 | 0.0707 | +18.84% |
| narrativeqa | ROUGE-L | 0.0574 | 0.0653 | +13.71% |
| qasper | F1-Score | 0.1311 | 0.1146 | -12.56% |
| qasper | ROUGE-L | 0.1233 | 0.1038 | -15.83% |

## 3) Critical Analysis

- Cross-Document Reasoning: GraphRAG outperformed baseline on 15 samples and underperformed on 15 samples (F1-based pairwise comparison across matched subset/sample IDs). Retrieval mode usage: local=31, global=9.
- Error Analysis: Error focus should target samples where baseline wins: inspect whether entity/relation extraction or community summaries omitted key facts, and whether local/global mode selection mismatched the question's reasoning depth.

## 4) Artifacts
github repo: https://github.com/ElkattoufiMohamed/graphrag_evaluation.git
