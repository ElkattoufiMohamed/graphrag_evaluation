# GraphRAG Evaluation Pipeline (Internship Guide Aligned)

This project evaluates **GraphRAG (nano-graphrag)** vs **Standard Vector RAG** on LongBench subsets:
- `musique`
- `2wikimqa` (WikiMQA config name in LongBench)
- `narrativeqa`
- `qasper`

Each subset uses **top 10 test samples**. The 10 contexts are aggregated into one unified retrieval pool per subset.

## Fixed Settings (Guide)
- Chunk size: `512`
- Chunk overlap: `50`
- Retrieval top-k: `10`
- Metrics: `F1-Score`, `ROUGE-L`
- GraphRAG retrieval mode: `auto` (chooses `local`/`global` per question heuristic)

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/run_experiment.sh
```

## Outputs
- `outputs/baseline_predictions.jsonl`
- `outputs/graphrag_predictions.jsonl`
- `outputs/performance_comparison.csv`
- `outputs/final_report.md`

Token-consumption fields are recorded in prediction JSONL files (embedding/index and LLM usage) and rolled up in the final report table.

See `docs/EXPERIMENT_GUIDE.md` for full step-by-step instructions.
