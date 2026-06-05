import os
from src.dataset.quality_check.llm_judge import LLMJudge


def run_judge(question: str, document_path: str, confidence_threshold: float) -> dict:
    judge = LLMJudge(
        model=os.getenv("JUDGE_MODEL", "gemini-2.0-flash"),
        confidence_threshold=confidence_threshold,
        base_url=os.getenv("JUDGE_BASE_URL") or None,
        max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "3")),
        max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "2048")),
    )
    result = judge.evaluate(question, document_path)
    return {
        "verdict": result.verdict,
        "confidence": result.confidence,
        "reason": result.reason,
        "suggested_question": result.suggested_question,
    }
