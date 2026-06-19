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

import itertools
import threading

from .base_model import BaseVisionModel, PredictionResult
from .inference_utils import page_to_b64 as _load_image_b64
from .inference_utils import parse_unanswerable as _parse_unanswerable


class VllmModel(BaseVisionModel):
    """Vision LLM backend powered by one or more vLLM servers.

    Args:
        base_url: vLLM server base URL, or a list of base URLs (must each expose
            /v1/chat/completions). When a list is given, requests are round-robined
            across the servers — useful for spreading concurrent requests across
            multiple single-GPU vLLM replicas (data parallelism) instead of a single
            multi-GPU tensor-parallel instance.
        model_id: HuggingFace model ID to pass in API requests.
        api_key: Optional auth token; vLLM accepts any non-empty string locally.
        max_tokens: Maximum tokens in the completion.
    """

    def __init__(
        self,
        base_url: str | list[str] = "http://localhost:8083/v1",
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
        urls = base_url if isinstance(base_url, list) else [base_url]
        self._urls = [u.rstrip("/") + "/chat/completions" for u in urls]
        self._url_cycle = itertools.cycle(self._urls)
        self._url_lock = threading.Lock()
        self._model_id = model_id
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._image_placeholder = image_placeholder
        self._max_image_pixels = max_image_pixels
        self._stop_sequences = stop_sequences or []
        self._stop_token_ids = stop_token_ids or []

    def _next_url(self) -> str:
        with self._url_lock:
            return next(self._url_cycle)

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
        try:
            resp = self._requests.post(self._next_url(), json=payload, headers=headers, timeout=50)
            if not resp.ok:
                detail = resp.text[:300]
                raise self._requests.exceptions.HTTPError(detail, response=resp)
        except self._requests.exceptions.RequestException as exc:
            return PredictionResult(
                sample_id="",
                predicted_unanswerable=False,
                confidence=0.0,
                raw_response=f"[SKIPPED: request failed] {exc}",
                skipped=True,
            )
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
        try:
            resp = self._requests.post(self._next_url(), json=payload, headers=headers, timeout=10)
            if not resp.ok:
                detail = resp.text[:300]
                raise self._requests.exceptions.HTTPError(detail, response=resp)
        except self._requests.exceptions.RequestException as exc:
            return f"[SKIPPED: request failed] {exc}"
        return resp.json()["choices"][0]["message"]["content"]
