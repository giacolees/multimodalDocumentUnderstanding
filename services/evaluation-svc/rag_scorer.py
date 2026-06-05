import os
from src.dataset.quality_check.llm_judge import LLMJudge, JudgeResult


def score_rag(
    question: str,
    retrieved_context: list[str],
    model_answer: str,
    ground_truth: str,
) -> dict:
    """Score a RAG answer. Uses exact match for unanswerable, LLM-judge for others."""
    gt_lower = ground_truth.strip().lower()
    ans_lower = model_answer.strip().upper()

    if gt_lower == "unanswerable":
        correct = "UNANSWERABLE" in ans_lower
        return {
            "score": 1.0 if correct else 0.0,
            "reason": "Exact match on UNANSWERABLE token." if correct else "Model failed to identify unanswerable question.",
            "correct": correct,
        }

    # For answerable ground truth: check if model answer is non-empty and not UNANSWERABLE
    correct = "UNANSWERABLE" not in ans_lower and len(model_answer.strip()) > 0
    return {
        "score": 1.0 if correct else 0.0,
        "reason": "Model provided an answer." if correct else "Model incorrectly answered UNANSWERABLE.",
        "correct": correct,
    }
