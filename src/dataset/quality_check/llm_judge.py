"""LLM-as-a-judge: verifies that a corrupted question is genuinely unanswerable
given its source document, and suggests a better corruption when it is not.

Uses pydantic-ai for structured output — no manual JSON parsing.
Supports two backends:
  - Gemini (cloud): pass model="gemini-2.0-flash", no base_url (needs GEMINI_API_KEY)
  - llama.cpp server: pass base_url="http://localhost:8080/v1" and model=<model-id>
    (OpenAI-compatible endpoint; start with `llama-server -hf <model>`)
"""

import os
import re
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.output import TextOutput


# ---------------------------------------------------------------------------
# Output schema — pydantic-ai enforces this against the model response
# ---------------------------------------------------------------------------

class JudgeResult(BaseModel):
    verdict: Literal["unanswerable", "answerable"]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# Hardcoded agent — model, prompt, and output type are fixed by design
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert annotator for Document Question Answering benchmarks.
You are given an image of a document page and a question. Decide whether
the question is answerable or unanswerable strictly from what is visible
in the document.

Think step-by-step before reaching your verdict:
  1. What relevant facts, numbers, dates, names, or entities are visible in the document?
  2. What exactly does the question ask for?
  3. Is that specific information present in the document?

Respond ONLY with a valid JSON object. No prose, no markdown fences. Required fields:
  verdict    – "unanswerable" or "answerable"
  confidence – float 0.0–1.0; use ≥0.85 only when certain, 0.5–0.7 for borderline

Examples:
  Question: "What was the total revenue in 1987?" — document shows 1992 figures only.
  → {"verdict": "unanswerable", "confidence": 0.9}

  Question: "What was the total revenue in 1992?" — document shows the 1992 revenue clearly.
  → {"verdict": "answerable", "confidence": 0.85}\
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Text-mode parser used for llama.cpp / OpenRouter backend
# ---------------------------------------------------------------------------

def _parse_judge_result(text: str) -> JudgeResult:
    """Extract verdict and confidence from free-form model text via regex."""
    # Strip thinking blocks and markdown
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    verdict_match = re.search(r"\b(unanswerable|answerable)\b", text, re.IGNORECASE)
    verdict = verdict_match.group(1).lower() if verdict_match else "answerable"

    conf_match = re.search(r'"?confidence"?\s*[=:]\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    return JudgeResult(verdict=verdict, confidence=min(max(confidence, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

_FALLBACK_MODEL = "gemini-2.0-flash"
_FALLBACK_CONFIDENCE_THRESHOLD = 0.5
_FALLBACK_MAX_RETRIES = 3
_FALLBACK_MAX_TOKENS = 2048


class LLMJudge:
    """Thin wrapper around a pydantic-ai judge agent.

    When base_url is provided the judge uses the OpenAI-compatible endpoint
    (e.g. llama-server at http://localhost:8080/v1).  Otherwise it falls back
    to Gemini via the Google provider (GEMINI_API_KEY required).
    """

    def __init__(
        self,
        model: str = _FALLBACK_MODEL,
        confidence_threshold: float = _FALLBACK_CONFIDENCE_THRESHOLD,
        base_url: Optional[str] = None,
        max_retries: int = _FALLBACK_MAX_RETRIES,
        max_tokens: int = _FALLBACK_MAX_TOKENS,
    ):
        self.confidence_threshold = confidence_threshold
        self._model = model
        self._base_url = base_url
        self._max_retries = max_retries
        self._max_tokens = max_tokens
        print(f"Initialized LLMJudge with model={model}, base_url={base_url}, confidence_threshold={confidence_threshold}, max_retries={max_retries}, max_tokens={max_tokens}")

    def _build_agent(self) -> "Agent[None, JudgeResult]":
        if self._base_url:
            # llama.cpp server (or any OpenAI-compatible endpoint).
            # TextOutput + custom parser: model returns plain text, we extract
            # JSON client-side — avoids tool calls (server-side JSON validation)
            # and avoids PromptedOutput (which injects a mid-conversation system
            # message that Qwen's Jinja template rejects).
            from pydantic_ai.models.openai import OpenAIModel  # type: ignore[import-untyped]
            from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore[import-untyped]
            pydantic_model = OpenAIModel(
                self._model,
                provider=OpenAIProvider(
                    base_url=self._base_url,
                    api_key=os.getenv("JUDGE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "sk-no-key",
                ),
            )
            output_type = TextOutput(_parse_judge_result)
        else:
            # Gemini cloud — model string is passed directly (pydantic-ai resolves it)
            pydantic_model = self._model  # type: ignore[assignment]
            output_type = JudgeResult  # type: ignore[assignment]

        return Agent(
            model=pydantic_model,
            output_type=output_type,
            system_prompt=_SYSTEM,
            retries=self._max_retries,
        )

    def evaluate(self, question: str, document_path: str) -> JudgeResult:
        """Return a JudgeResult for *question* against the given document image.

        When the verdict is "answerable", suggested_question may contain a
        judge-authored rewrite worth a second evaluation pass.
        """
        if not hasattr(self, "_agent"):
            self._agent = self._build_agent()
        image_bytes, media_type = _read_image(document_path)
        result = self._agent.run_sync(
            [
                BinaryContent(data=image_bytes, media_type=media_type),
                f"Question: {question}\n\nIs this question answerable from the document shown?",
            ],
            model_settings={"max_tokens": self._max_tokens},
        )
        jr = result.output
        # Demote low-confidence unanswerable verdicts so borderline samples are dropped
        if jr.verdict == "unanswerable" and jr.confidence < self.confidence_threshold:
            jr.verdict = "answerable"
        return jr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_image(image_path: str) -> tuple[bytes, str]:
    path = Path(image_path)
    ext = path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }
    media_type = media_map.get(ext, "image/png")
    return path.read_bytes(), media_type
