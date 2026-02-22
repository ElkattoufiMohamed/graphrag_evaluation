#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${EVAL_LLM_PROVIDER:-qwen}" == "qwen" && -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "[ERROR] EVAL_LLM_PROVIDER=qwen but DASHSCOPE_API_KEY is not set"
  exit 1
fi

if [[ "${EVAL_LLM_PROVIDER:-qwen}" == "qwen" && -z "${DASHSCOPE_BASE_HTTP_API_URL:-}" ]]; then
  echo "[WARN] DASHSCOPE_BASE_HTTP_API_URL not set; defaulting to DashScope intl endpoint inside code"
fi

if [[ "${EVAL_LLM_PROVIDER:-qwen}" == "gemini" && -z "${GEMINI_API_KEY:-}" ]]; then
  echo "[ERROR] EVAL_LLM_PROVIDER=gemini but GEMINI_API_KEY is not set"
  exit 1
fi

if [[ "${EVAL_LLM_PROVIDER:-qwen}" == "ollama" ]]; then
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
  if ! curl -sS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "[ERROR] EVAL_LLM_PROVIDER=ollama but Ollama server is not reachable at $OLLAMA_BASE_URL"
    echo "        Start it first (e.g., 'ollama serve') and confirm the model is pulled."
    exit 1
  fi
fi

if [[ "${EVAL_EMBEDDING_BACKEND:-st}" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] EVAL_EMBEDDING_BACKEND=openai but OPENAI_API_KEY is not set"
  exit 1
fi

echo "[1/4] Running baseline..."
python -m src.run_baseline

echo "[2/4] Running GraphRAG..."
python -m src.run_graphrag

echo "[3/4] Computing comparison matrix..."
python -m src.eval_compare

echo "[4/4] Building final report..."
python -m src.build_report

echo "Done. Check outputs/ directory."
