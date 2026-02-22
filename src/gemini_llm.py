from __future__ import annotations
import os
from google import genai


class GeminiLLM:
    def __init__(self, model: str = "gemini-3-pro-preview"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY env var.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # Deterministic-ish: temperature=0
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            # New SDK supports generation config; keep minimal for compatibility
            # If your SDK version supports config=..., you can add it.
        )
        usage_meta = getattr(resp, "usage_metadata", None)
        if usage_meta is not None:
            self.last_usage = {
                "prompt_tokens": int(getattr(usage_meta, "prompt_token_count", 0) or 0),
                "completion_tokens": int(getattr(usage_meta, "candidates_token_count", 0) or 0),
                "total_tokens": int(getattr(usage_meta, "total_token_count", 0) or 0),
            }
        return (resp.text or "").strip()
