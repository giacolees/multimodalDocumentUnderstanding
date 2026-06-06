import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Load main.py under a unique module name to avoid sys.modules['main'] collisions
# when all three service test suites run in the same pytest process.
_spec = importlib.util.spec_from_file_location(
    "evaluation_svc_main",
    os.path.join(os.path.dirname(__file__), "..", "main.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["evaluation_svc_main"] = _mod
_spec.loader.exec_module(_mod)
app = _mod.app

client = TestClient(app)


def test_metrics_precision_recall_f1():
    """2 TP, 1 FP, 1 FN, 1 TN → precision=2/3, recall=2/3, f1=2/3."""
    resp = client.post("/evaluate/metrics", json={
        "y_true": [True, True, False, True, False],
        "y_pred": [True, True, True, False, False],
    })
    assert resp.status_code == 200
    d = resp.json()
    assert d["tp"] == 2
    assert d["fp"] == 1
    assert d["fn"] == 1
    assert d["tn"] == 1
    assert abs(d["precision"] - 2/3) < 0.01
    assert abs(d["recall"] - 2/3) < 0.01
    assert abs(d["f1"] - 2/3) < 0.01


def test_answerability_judge_called_with_correct_args():
    mock_result = {
        "verdict": "unanswerable",
        "confidence": 0.9,
        "reason": "Date not in document.",
        "suggested_question": None,
    }
    with patch("judge.run_judge", return_value=mock_result) as mock_judge:
        resp = client.post("/evaluate/answerability", json={
            "question": "What year?",
            "document_path": "data/raw/doc.png",
            "confidence_threshold": 0.5,
        })
    assert resp.status_code == 200
    assert resp.json()["verdict"] == "unanswerable"
    assert resp.json()["confidence"] == 0.9
    mock_judge.assert_called_once_with("What year?", "data/raw/doc.png", 0.5)


def test_answerability_judge_exception_returns_null_verdict():
    """Per spec: judge failures return null verdict, don't crash."""
    with patch("judge.run_judge", side_effect=Exception("API timeout")):
        resp = client.post("/evaluate/answerability", json={
            "question": "What year?",
            "document_path": "missing.png",
            "confidence_threshold": 0.5,
        })
    assert resp.status_code == 200
    assert resp.json()["verdict"] is None
    assert resp.json()["confidence"] == 0.0


def test_rag_correct_unanswerable():
    """Model correctly answers UNANSWERABLE when ground truth is unanswerable."""
    with patch("rag_scorer.score_rag", return_value={
        "score": 1.0, "reason": "Correct.", "correct": True
    }):
        resp = client.post("/evaluate/rag", json={
            "question": "What is the 1987 revenue?",
            "retrieved_context": ["Revenue 2019: $1M"],
            "model_answer": "UNANSWERABLE",
            "ground_truth": "unanswerable",
        })
    assert resp.status_code == 200
    assert resp.json()["correct"] is True
    assert resp.json()["score"] == 1.0


def test_rag_scorer_unanswerable_correct():
    from rag_scorer import score_rag
    result = score_rag("What year?", ["Revenue 2019"], "UNANSWERABLE", "unanswerable")
    assert result["correct"] is True
    assert result["score"] == 1.0


def test_rag_scorer_unanswerable_incorrect():
    from rag_scorer import score_rag
    result = score_rag("What year?", ["Revenue 2019"], "The year is 2019", "unanswerable")
    assert result["correct"] is False
    assert result["score"] == 0.0


def test_rag_scorer_answerable_correct():
    from rag_scorer import score_rag
    result = score_rag("What year?", ["Revenue 2019"], "The answer is 2019", "2019")
    assert result["correct"] is True
    assert result["score"] == 1.0
