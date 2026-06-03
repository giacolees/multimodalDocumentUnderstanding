"""LLM-as-a-judge: verifies that a corrupted question is genuinely unanswerable
given its source document, and suggests a better corruption when it is not.

Uses pydantic-ai for structured output — no manual JSON parsing.
Model: Gemini 2.0 Flash (multimodal, GOOGLE_API_KEY).
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.gemini import GeminiModel  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Output schema — pydantic-ai enforces this against the model response
# ---------------------------------------------------------------------------

class JudgeResult(BaseModel):
    verdict: Literal["unanswerable", "answerable"]
    reason: str
    # Populated only when verdict="answerable": a minimal rewrite the judge
    # believes would make the question unanswerable from this document.
    suggested_question: Optional[str] = None


# ---------------------------------------------------------------------------
# Hardcoded agent — model, prompt, and output type are fixed by design
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert annotator for Document Question Answering benchmarks.
You are given an image of a document page and a question. Decide whether
the question is answerable or unanswerable strictly from what is visible
in the document.

Fill in every field of the structured response:
  verdict            – "unanswerable" or "answerable"
  reason             – one concise sentence explaining your decision
  suggested_question – if verdict is "answerable", provide the minimal rewrite
                       of the question that makes it unanswerable (swap an
                       entity, date, or value that is absent from the document).
                       Must be fluent and plausible. Set to null otherwise.\
"""

_judge_agent: Agent[None, JudgeResult] = Agent(
    model=GeminiModel("gemini-2.0-flash"),
    result_type=JudgeResult,
    system_prompt=_SYSTEM,
)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LLMJudge:
    """Thin wrapper around the pydantic-ai judge agent."""

    def evaluate(self, question: str, document_path: str) -> JudgeResult:
        """Return a JudgeResult for *question* against the given document image.

        When the verdict is "answerable", suggested_question may contain a
        judge-authored rewrite worth a second evaluation pass.
        """
        image_bytes, media_type = _read_image(document_path)
        result = _judge_agent.run_sync(
            [
                BinaryContent(data=image_bytes, media_type=media_type),
                f"Question: {question}\n\nIs this question answerable from the document shown?",
            ],
            model_settings={"max_tokens": 512},
        )
        jr = result.output
        # Enforce invariant: no suggestion when already unanswerable
        if jr.verdict == "unanswerable":
            jr.suggested_question = None
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
