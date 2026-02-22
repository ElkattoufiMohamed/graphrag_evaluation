from __future__ import annotations

import os
import time
import requests


class OllamaLLM:
    def __init__(self, model: str = "qcwind/qwen2.5-7B-instruct-Q4_K_M"):
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.timeout_s = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "1800"))
        self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": float(temperature),
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }

        resp = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=(10, self.timeout_s))
                break
            except requests.exceptions.ReadTimeout:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Ollama request timed out after {self.timeout_s}s (retries={self.max_retries}). "
                        "Try increasing OLLAMA_TIMEOUT_SECONDS, lowering OLLAMA_NUM_PREDICT, "
                        "or using a smaller/faster model."
                    )
                time.sleep(1.5 * (attempt + 1))

        if resp is None or resp.status_code != 200:
            code = resp.status_code if resp is not None else "N/A"
            text = resp.text if resp is not None else "No response"
            raise RuntimeError(f"Ollama error {code}: {text}")

        data = resp.json()
        # Ollama returns eval/prompt counts in various fields depending on version.
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        message = data.get("message", {})
        return str(message.get("content", "")).strip()
