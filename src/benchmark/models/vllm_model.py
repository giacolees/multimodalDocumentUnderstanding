"""vLLM backend for BaseVisionModel.

Talks to a running vLLM server via its OpenAI-compatible /v1/chat/completions
endpoint.  Start a server with:

    vllm serve google/gemma-4-12b-it --port 8083

Config example (benchmark_config.yaml):

  - backend: vllm
    base_url: http://localhost:8083/v1
    model_id: google/gemma-4-12b-it
    api_key: ""          # optional; vLLM accepts any non-empty string locally
"""

from __future__ import annotations

import base64
from pathlib import Path

from .base_model import BaseVisionModel, PredictionResult


def _load_image_b64(document_path: str, page_index: int = 0, max_pixels: int = 0) -> str:
    import io
    from PIL import Image
    path = Path(document_path)
    if path.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise ImportError("pdf2image is required for PDF documents.") from exc
        pages = convert_from_path(str(path), first_page=page_index + 1, last_page=page_index + 1)
        if not pages:
            raise ValueError(f"No page {page_index} in {path}")
        img = pages[0]
    else:
        img = Image.open(path)

    if max_pixels > 0:
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _parse_unanswerable(text: str | None) -> tuple[bool, float]:
    if not text:
        return False, 0.0
    upper = text.upper()
    if "UNANSWERABLE" in upper:
        return True, 0.9
    negative_phrases = ["cannot be answered", "not in the document", "no information",
                        "not mentioned", "not found", "cannot answer", "not provided"]
    if any(p in upper for p in (p.upper() for p in negative_phrases)):
        return True, 0.7
    return False, 0.1


class VllmModel(BaseVisionModel):
    """Vision LLM backend powered by a vLLM server.

    Args:
        base_url: vLLM server base URL (must expose /v1/chat/completions).
        model_id: HuggingFace model ID to pass in API requests.
        api_key: Optional auth token; vLLM accepts any non-empty string locally.
        max_tokens: Maximum tokens in the completion.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8083/v1",
        model_id: str = "google/gemma-4-12b-it",
        api_key: str = "local",
        max_tokens: int = 256,
        image_placeholder: str = "",
        max_image_pixels: int = 0,
        stop_sequences: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        import requests
        self._requests = requests
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model_id = model_id
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._image_placeholder = image_placeholder
        self._max_image_pixels = max_image_pixels
        self._stop_sequences = stop_sequences or []
        self._stop_token_ids = stop_token_ids or []

    def name(self) -> str:
        return f"vllm:{self._model_id}"

    def predict_unanswerable(
        self,
        document_path: str,
        question: str,
        prompt_template: str,
        page_indices: list[int] | None = None,
    ) -> PredictionResult:
        page = (page_indices or [0])[0]
        image_b64 = _load_image_b64(document_path, page_index=page, max_pixels=self._max_image_pixels)
        prompt = prompt_template.format(question=question)

        text = f"{self._image_placeholder}\n{prompt}" if self._image_placeholder else prompt
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": text},
        ]
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        payload: dict = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
        }
        if self._stop_sequences:
            payload["stop"] = self._stop_sequences
        if self._stop_token_ids:
            payload["stop_token_ids"] = self._stop_token_ids
        resp = self._requests.post(self._url, json=payload, headers=headers, timeout=120)
        if not resp.ok:
            detail = resp.text[:300]
            if resp.status_code == 400 and "exceeds" in detail and "context" in detail:
                return PredictionResult(
                    sample_id="",
                    predicted_unanswerable=False,
                    confidence=0.0,
                    raw_response=f"[SKIPPED: context too long] {detail}",
                )
            resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        predicted, confidence = _parse_unanswerable(raw)
        return PredictionResult(
            sample_id="",
            predicted_unanswerable=predicted,
            confidence=confidence,
            raw_response=raw,
        )

    def generate(
        self,
        document_path: str,
        prompt: str,
        page_index: int | None = None,
        max_tokens: int = 1024,
    ) -> str:
        page = page_index if page_index is not None else 0
        image_b64 = _load_image_b64(document_path, page_index=page,
                                    max_pixels=self._max_image_pixels)
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        payload: dict = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        resp = self._requests.post(self._url, json=payload, headers=headers, timeout=120)
        if not resp.ok:
            detail = resp.text[:300]
            if resp.status_code == 400 and "exceeds" in detail and "context" in detail:
                return f"[SKIPPED: context too long] {detail}"
            resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
