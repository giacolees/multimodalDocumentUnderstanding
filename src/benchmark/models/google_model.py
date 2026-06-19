"""Google Gemini backend for BaseVisionModel.

Uses the Gemini REST API directly (no extra SDK dependency).
Vision-capable models: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash.

Required env var: GEMINI_API_KEY

Config example (benchmark_config.yaml):
  - backend: google
    model_id: gemini-2.0-flash
"""

from __future__ import annotations

import os

import requests

from .base_model import BaseVisionModel, PredictionResult
from .inference_utils import page_to_b64 as _page_to_b64
from .inference_utils import parse_unanswerable as _parse_unanswerable

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GoogleModel(BaseVisionModel):
    def __init__(self, model_id: str = "gemini-2.0-flash") -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        self._model_id = model_id
        self._api_key = api_key

    def name(self) -> str:
        return f"google:{self._model_id}"

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
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": b64}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {"maxOutputTokens": 256, "temperature": 0.0},
        }
        url = _BASE_URL.format(model=self._model_id)
        resp = requests.post(url, json=payload, params={"key": self._api_key}, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        predicted, confidence = _parse_unanswerable(raw)
        return PredictionResult(
            sample_id="",
            predicted_unanswerable=predicted,
            confidence=confidence,
            raw_response=raw,
        )
