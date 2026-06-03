"""llama.cpp backend for BaseVisionModel.

Two operating modes selected by the `mode` constructor argument:

  "direct"  – uses the llama-cpp-python binding to run a local GGUF model in-process.
               Requires `pip install llama-cpp-python` (with METAL/CUDA extras as needed).
               Vision support requires a companion CLIP projector (clip_model_path).

  "server"  – sends requests to a running llama.cpp HTTP server
               (`llama-server --mmproj … --model …`), which exposes an
               OpenAI-compatible /v1/chat/completions endpoint.
               No extra Python package is needed beyond `requests`.

Config example (benchmark_config.yaml):

  - backend: llama_cpp
    mode: direct
    model_path: models/llava-v1.6-mistral-7b.Q4_K_M.gguf
    clip_model_path: models/llava-v1.6-mistral-7b-mmproj.gguf
    n_ctx: 4096
    n_gpu_layers: -1       # -1 = offload all layers to GPU

  - backend: llama_cpp
    mode: server
    base_url: http://localhost:8080/v1
    model_id: llava-v1.6-mistral-7b   # passed in the API request; many servers ignore it
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from .base_model import BaseVisionModel, PredictionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_b64(document_path: str, page_index: int = 0) -> str:
    """Return a base64-encoded PNG string for the requested page/image."""
    path = Path(document_path)

    if path.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise ImportError("pdf2image is required for PDF documents.") from exc
        pages = convert_from_path(str(path), first_page=page_index + 1, last_page=page_index + 1)
        if not pages:
            raise ValueError(f"No page {page_index} in {path}")
        import io
        buf = io.BytesIO()
        pages[0].save(buf, format="PNG")
        raw = buf.getvalue()
    else:
        raw = path.read_bytes()

    return base64.standard_b64encode(raw).decode()


def _parse_unanswerable(text: str) -> tuple[bool, float]:
    """Heuristic: detect UNANSWERABLE signal in model output."""
    upper = text.upper()
    if "UNANSWERABLE" in upper:
        return True, 0.9
    negative_phrases = ["cannot be answered", "not in the document", "no information",
                        "not mentioned", "not found", "cannot answer", "not provided"]
    if any(p in upper for p in (p.upper() for p in negative_phrases)):
        return True, 0.7
    return False, 0.1


# ---------------------------------------------------------------------------
# Direct binding (llama-cpp-python)
# ---------------------------------------------------------------------------

class _DirectBackend:
    def __init__(
        self,
        model_path: str,
        clip_model_path: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python"
            ) from exc

        chat_handler = None
        if clip_model_path:
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path, verbose=False)
            except Exception as exc:
                raise RuntimeError(f"Failed to load CLIP model at {clip_model_path}: {exc}") from exc

        self._llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._has_vision = clip_model_path is not None

    def complete(self, prompt: str, image_b64: str | None) -> str:
        content: list[dict] = []
        if image_b64 and self._has_vision:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            })
        content.append({"type": "text", "text": prompt})

        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Server mode (HTTP OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

class _ServerBackend:
    def __init__(self, base_url: str, model_id: str = "local") -> None:
        import requests  # already in project deps
        self._requests = requests
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model_id = model_id

    def complete(self, prompt: str, image_b64: str | None) -> str:
        content: list[dict] = []
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            })
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        resp = self._requests.post(self._url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

class LlamaCppModel(BaseVisionModel):
    """Vision LLM backend powered by llama.cpp.

    Args:
        mode: "direct" (in-process via llama-cpp-python) or "server" (HTTP).
        model_path: Path to the GGUF model file (direct mode only).
        clip_model_path: Path to the CLIP projector GGUF (direct mode, optional).
        n_ctx: Context size in tokens (direct mode).
        n_gpu_layers: GPU layers to offload; -1 = all (direct mode).
        base_url: llama.cpp server base URL (server mode).
        model_id: Model identifier sent in API requests (server mode).
    """

    def __init__(
        self,
        mode: str = "server",
        model_path: str | None = None,
        clip_model_path: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        base_url: str = "http://localhost:8080/v1",
        model_id: str = "local",
    ) -> None:
        if mode == "direct":
            if not model_path:
                raise ValueError("model_path is required for mode='direct'")
            self._backend = _DirectBackend(
                model_path=model_path,
                clip_model_path=clip_model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
            )
            self._model_label = Path(model_path).stem
        elif mode == "server":
            self._backend = _ServerBackend(base_url=base_url, model_id=model_id)
            self._model_label = f"llama_cpp_server:{model_id}"
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'direct' or 'server'.")

    # ------------------------------------------------------------------

    def name(self) -> str:
        return f"llama_cpp:{self._model_label}"

    def predict_unanswerable(
        self,
        document_path: str,
        question: str,
        prompt_template: str,
        page_indices: list[int] | None = None,
    ) -> PredictionResult:
        page = (page_indices or [0])[0]
        image_b64 = _load_image_b64(document_path, page_index=page)
        prompt = prompt_template.format(question=question)
        raw = self._backend.complete(prompt, image_b64)
        predicted, confidence = _parse_unanswerable(raw)
        return PredictionResult(
            sample_id="",
            predicted_unanswerable=predicted,
            confidence=confidence,
            raw_response=raw,
        )
