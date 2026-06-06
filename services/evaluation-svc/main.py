import sys, os
sys.path.insert(0, "/app")

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

import judge as judge_module
import rag_scorer
from shared.observability import setup_tracing, setup_metrics, get_logger

app = FastAPI(title="evaluation-svc", version="1.0")
setup_tracing("evaluation-svc")
setup_metrics(app)
logger = get_logger("evaluation-svc")


class AnswerabilityRequest(BaseModel):
    question: str
    document_path: str
    confidence_threshold: float = 0.5


class AnswerabilityResponse(BaseModel):
    verdict: Optional[str]
    confidence: float
    reason: str
    suggested_question: Optional[str] = None


@app.post("/evaluate/answerability", response_model=AnswerabilityResponse)
def evaluate_answerability(req: AnswerabilityRequest):
    try:
        result = judge_module.run_judge(req.question, req.document_path, req.confidence_threshold)
        return AnswerabilityResponse(**result)
    except Exception as e:
        logger.warning("Judge call failed", extra={"error": str(e)})
        return AnswerabilityResponse(verdict=None, confidence=0.0, reason=str(e))


class RAGEvalRequest(BaseModel):
    question: str
    retrieved_context: list[str]
    model_answer: str
    ground_truth: str


class RAGEvalResponse(BaseModel):
    score: float
    reason: str
    correct: bool


@app.post("/evaluate/rag", response_model=RAGEvalResponse)
def evaluate_rag(req: RAGEvalRequest):
    try:
        result = rag_scorer.score_rag(
            req.question, req.retrieved_context, req.model_answer, req.ground_truth
        )
        return RAGEvalResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MetricsRequest(BaseModel):
    y_true: list[bool]
    y_pred: list[bool]

    @model_validator(mode="after")
    def lengths_must_match(self):
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"y_true length ({len(self.y_true)}) must equal y_pred length ({len(self.y_pred)})"
            )
        return self


class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


@app.post("/evaluate/metrics", response_model=MetricsResponse)
def evaluate_metrics(req: MetricsRequest):
    try:
        from src.benchmark.evaluation.metrics import compute_metrics
        m = compute_metrics(req.y_true, req.y_pred)
        return MetricsResponse(**m.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
