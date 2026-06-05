def score_rag(
    question: str,
    retrieved_context: list[str],
    model_answer: str,
    ground_truth: str,
) -> dict:
    """Score a RAG answer. Uses exact match for unanswerable, substring match for others."""
    gt_lower = ground_truth.strip().lower()
    ans_upper = model_answer.strip().upper()

    if gt_lower == "unanswerable":
        correct = "UNANSWERABLE" in ans_upper
        return {
            "score": 1.0 if correct else 0.0,
            "reason": "Exact match on UNANSWERABLE token." if correct else "Model failed to identify unanswerable question.",
            "correct": correct,
        }

    # For answerable ground truth: check ground_truth appears in model answer
    correct = (
        "UNANSWERABLE" not in ans_upper
        and ground_truth.strip().lower() in model_answer.strip().lower()
    )
    return {
        "score": 1.0 if correct else 0.0,
        "reason": "Answer matches ground truth." if correct else "Answer does not match ground truth or model answered UNANSWERABLE.",
        "correct": correct,
    }
