"""Thin wrappers around the two LLM back-ends used by the pipeline.

* `HFChatGenerator`: Hugging Face Transformers pipeline (Class 3, Qwen2.5-14B-Instruct).
* `OllamaClient`:    local Ollama server (Class 4 gemma4:31b, Class 5 gemma4:26b).
"""

from __future__ import annotations

import requests


class HFChatGenerator:
    """Chat-template text generation with a local Transformers model (bf16, auto device map)."""

    def __init__(self, model_name: str, max_new_tokens: int = 256):
        import torch
        from transformers import pipeline

        self.max_new_tokens = max_new_tokens
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def generate(self, system: str, user: str, **sampling) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        tokenizer = self.pipe.tokenizer
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            **sampling,
        )
        return output[0]["generated_text"][len(prompt):].strip()


class OllamaClient:
    """Non-streaming calls to the Ollama `/api/generate` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.options = {"temperature": temperature, "top_p": top_p}

    def check_available(self) -> None:
        """Exit with a clear message if the server is unreachable or the model is not pulled."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(f"Cannot reach Ollama at {self.base_url}: {exc}") from None
        models = {m.get("name") for m in response.json().get("models", [])}
        if self.model not in models:
            raise SystemExit(
                f"Model '{self.model}' is not available at {self.base_url}. "
                f"Run `ollama pull {self.model}` (available: {sorted(models)})."
            )

    def generate(self, prompt: str) -> str | None:
        """Return the generated text, or None on any failure or empty response.

        Failures are never replaced by the input text, so a failed call can simply be
        retried by re-running the script.
        """
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": self.options}
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            text = response.json().get("response", "").strip()
        except (requests.RequestException, ValueError):
            return None
        return text or None
