"""OpenRouter backend for BaseVisionModel.

OpenRouter proxies requests to many providers using an OpenAI-compatible endpoint.
Pick any vision-capable model via model_id (e.g. google/gemini-2.0-flash-exp,
meta-llama/llama-3.2-90b-vision-instruct, mistralai/pixtral-12b).

Required env var: OPENROUTER_API_KEY

Config example (benchmark_config.yaml):
  - backend: openrouter
    model_id: google/gemini-2.0-flash-exp
    site_url: https://your-app.example.com   # optional, shown in OR dashboard
    site_name: MyBenchmark                   # optional
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import requests

from .base_model import BaseVisionModel, PredictionResult


def _page_to_b64(document_path: str, page_index: int = 0) -> str:
    path = Path(document_path)
    if path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(str(path), first_page=page_index + 1, last_page=page_index + 1)
        if not pages:
            raise ValueError(f"No page {page_index} in {path}")
        buf = io.BytesIO()
        pages[0].save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode()
    return base64.standard_b64encode(path.read_bytes()).decode()


def _parse_unanswerable(text: str) -> tuple[bool, float]:
    upper = text.upper()
    if "UNANSWERABLE" in upper:
        return True, 0.9
    phrases = ["cannot be answered", "not in the document", "no information",
               "not mentioned", "not found", "cannot answer", "not provided"]
    if any(p.upper() in upper for p in phrases):
        return True, 0.7
    return False, 0.1


class OpenRouterModel(BaseVisionModel):
    _API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model_id: str = "google/gemini-2.0-flash-exp",
        site_url: str = "",
        site_name: str = "",
    ) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")
        self._model_id = model_id
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **({"HTTP-Referer": site_url} if site_url else {}),
            **({"X-Title": site_name} if site_name else {}),
        }

    def name(self) -> str:
        return f"openrouter:{self._model_id}"

    def predict_unanswerable(
        self,
        document_path: str,
        question: str,
        prompt_template: str,
        page_indices: list[int] | None = None,
    ) -> PredictionResult:
        page = (page_indices or [0])[0]
        b64 = _page_to_b64(document_path, page)
        prompt = prompt_template.format(question=question)

        payload = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        resp = requests.post(self._API_URL, json=payload, headers=self._headers, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        predicted, confidence = _parse_unanswerable(raw)
        return PredictionResult(
            sample_id="",
            predicted_unanswerable=predicted,
            confidence=confidence,
            raw_response=raw,
        )
