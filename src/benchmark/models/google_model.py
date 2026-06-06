"""Google Gemini backend for BaseVisionModel.

Uses the Gemini REST API directly (no extra SDK dependency).
Vision-capable models: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash.

Required env var: GEMINI_API_KEY

Config example (benchmark_config.yaml):
  - backend: google
    model_id: gemini-2.0-flash
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import requests

from .base_model import BaseVisionModel, PredictionResult

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


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
