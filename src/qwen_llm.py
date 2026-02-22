import os
import dashscope
from dashscope import Generation

class QwenLLM:
    def __init__(self, model: str = "qwen-plus"):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY env var.")
        dashscope.api_key = api_key
        # Region endpoint can differ from the account key region.
        # Default to intl (Singapore) per current docs.
        dashscope.base_http_api_url = os.getenv(
            "DASHSCOPE_BASE_HTTP_API_URL",
            "https://dashscope-intl.aliyuncs.com/api/v1",
        )
        self.model = model
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        resp = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=self.model,
            messages=messages,
            temperature=temperature,
            result_format="message",
        )
        if getattr(resp, "status_code", None) != 200:
            code = getattr(resp, "code", "")
            msg = getattr(resp, "message", str(resp))
            if code == "AccessDenied.Unpurchased":
                raise RuntimeError(
                    "DashScope AccessDenied.Unpurchased for model '%s'. "
                    "This is usually NOT a code bug: either your account is not entitled for this model, "
                    "or API key region and endpoint region don't match. "
                    "Check Model Studio entitlement and set DASHSCOPE_BASE_HTTP_API_URL correctly. "
                    "Raw message: %s" % (self.model, msg)
                )
            raise RuntimeError(f"DashScope error [{code}]: {msg}")

        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_toks = int(getattr(usage, "input_tokens", 0) or 0)
            completion_toks = int(getattr(usage, "output_tokens", 0) or 0)
            self.last_usage = {
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": prompt_toks + completion_toks,
            }
        output = getattr(resp, "output", None)
        if output is not None:
            choices = getattr(output, "choices", None) or []
            if choices:
                first = choices[0]
                message = getattr(first, "message", None)
                if isinstance(message, dict):
                    return str(message.get("content", "")).strip()
                if message is not None:
                    content = getattr(message, "content", "")
                    return str(content).strip()
        # fallback for older SDK response shapes
        return getattr(getattr(resp, "output", None), "text", "").strip()
