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

## Recommended Project Structure

```text
graphrag_eval/
├── src/                       # experiment code
├── scripts/
│   └── run_experiment.sh      # one-command full pipeline
├── docs/
│   └── EXPERIMENT_GUIDE.md    # detailed setup/run walkthrough
├── .env.example               # API key + config template
├── requirements.txt
└── outputs/                   # generated at runtime (gitignored)
```

## API Key Setup
1. Copy env template:
   ```bash
   cp .env.example .env
   ```
2. Fill keys based on your choices:
   - If `EVAL_LLM_PROVIDER=qwen`: set `DASHSCOPE_API_KEY`
     - Also set `DASHSCOPE_BASE_HTTP_API_URL` to the endpoint matching your key region
   - If `EVAL_LLM_PROVIDER=gemini`: set `GEMINI_API_KEY`
   - If `EVAL_LLM_PROVIDER=ollama`: set `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`) and run `ollama serve`
   - If `EVAL_EMBEDDING_BACKEND=openai`: set `OPENAI_API_KEY`

If you get `AccessDenied.Unpurchased` on `qwen-plus`, this is usually account entitlement or key-region mismatch (not pipeline logic).

### Run with local Ollama (your downloaded model)
```bash
export EVAL_LLM_PROVIDER=ollama
export EVAL_LLM_MODEL=qcwind/qwen2.5-7B-instruct-Q4_K_M
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_TIMEOUT_SECONDS=1800
export OLLAMA_MAX_RETRIES=3
export OLLAMA_NUM_PREDICT=512
export OLLAMA_NUM_CTX=8192
export GRAPHRAG_LLM_CONCURRENCY=1
export GRAPHRAG_JSON_REPAIR=1
./scripts/run_experiment.sh
```

If GraphRAG times out during entity extraction on local hardware, increase `OLLAMA_TIMEOUT_SECONDS` (e.g. 2400) and keep concurrency at `1`.
If GraphRAG fails on JSON parsing during community report generation, keep `GRAPHRAG_JSON_REPAIR=1` to auto-repair malformed local-model JSON.

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
