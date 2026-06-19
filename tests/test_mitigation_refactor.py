import pytest


def test_mitigation_strategy_is_abstract():
    """Cannot instantiate MitigationStrategy directly — build_prompt is abstract."""
    from src.mitigation.strategies.base import MitigationStrategy
    with pytest.raises(TypeError):
        MitigationStrategy()


def test_concrete_strategy_prepare_is_noop():
    """prepare() default implementation does nothing."""
    from src.mitigation.strategies.base import MitigationStrategy

    class Concrete(MitigationStrategy):
        name = "concrete"
        def build_prompt(self, item, model):
            return "prompt"

    s = Concrete()
    s.prepare([], None)  # must not raise


def test_few_shot_strategy_builds_prompt():
    from src.mitigation.strategies.few_shot import FewShotStrategy
    s = FewShotStrategy({"k": 2})
    item = {"corrupted_question": "What year?"}
    prompt = s.build_prompt(item, model=None)
    assert "UNANSWERABLE" in prompt
    assert "What year?" in prompt


def test_cot_strategy_builds_prompt():
    from src.mitigation.strategies.chain_of_thought import ChainOfThoughtStrategy
    s = ChainOfThoughtStrategy({})
    item = {"corrupted_question": "Where is Table 5?"}
    prompt = s.build_prompt(item, model=None)
    assert "step by step" in prompt.lower()
    assert "Where is Table 5?" in prompt


def test_knowledge_injection_strategy_builds_prompt():
    from src.mitigation.strategies.knowledge_injection import KnowledgeInjectionStrategy
    s = KnowledgeInjectionStrategy({})
    item = {"corrupted_question": "What year?"}
    prompt = s.build_prompt(item, model=None)
    assert "UNANSWERABLE" in prompt
    assert "What year?" in prompt


def test_evaluate_strategy_returns_metrics_and_logs_mlflow(tmp_path):
    import json
    import mlflow
    import unittest.mock as mock
    from src.benchmark.models.base_model import PredictionResult

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlruns.db")
    mlflow.set_experiment("mitigation")

    dataset = [
        {"sample_id": "s1", "document_path": "doc.png",
         "corrupted_question": "Q1?", "corruption_type": "nlp_entity"},
        {"sample_id": "s2", "document_path": "doc.png",
         "corrupted_question": "Q2?", "corruption_type": "element"},
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    mock_model = mock.MagicMock()
    mock_model.name.return_value = "test/model"
    mock_model.predict_unanswerable.return_value = PredictionResult(
        sample_id="", predicted_unanswerable=True, confidence=-1, raw_response="UNANSWERABLE"
    )

    class TrivialStrategy:
        name = "trivial"
        def prepare(self, dataset, model): pass
        def build_prompt(self, item, model): return "Is this unanswerable? Q: {question}"

    from src.mitigation.evaluation import evaluate_strategy
    result = evaluate_strategy(
        strategy=TrivialStrategy(),
        dataset=dataset,
        model=mock_model,
        baseline_metrics={"f1": 0.5, "mcc": 0.0, "precision": 0.5,
                          "recall": 0.5, "specificity": 0.5, "balanced_accuracy": 0.5},
        model_id="test/model",
        corrupted_dataset_path=str(dataset_path),
    )

    assert result["metrics"]["f1"] == 1.0
    assert len(result["records"]) == 2

    runs = mlflow.search_runs(experiment_names=["mitigation"], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["strategy"] == "trivial"
    assert run.data.params["model_id"] == "test/model"
    assert "f1" in run.data.metrics
    assert "mcc" in run.data.metrics
    assert "delta_f1" in run.data.metrics
    assert "delta_mcc" in run.data.metrics
    assert "f1_nlp_entity" in run.data.metrics


def test_inference_time_includes_build_prompt_duration(tmp_path):
    """inference_time_s must cover the whole per-sample pipeline (e.g. RAG retrieval
    inside build_prompt), not just the model call."""
    import json
    import time
    import mlflow
    import unittest.mock as mock
    from src.benchmark.models.base_model import PredictionResult

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlruns.db")
    mlflow.set_experiment("mitigation")

    dataset = [
        {"sample_id": "s1", "document_path": "doc.png",
         "corrupted_question": "Q1?", "corruption_type": "nlp_entity"},
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    mock_model = mock.MagicMock()
    mock_model.name.return_value = "test/model"
    mock_model.predict_unanswerable.return_value = PredictionResult(
        sample_id="", predicted_unanswerable=True, confidence=-1, raw_response="UNANSWERABLE"
    )

    SLEEP_S = 0.2

    class SlowBuildPromptStrategy:
        name = "slow"
        def prepare(self, dataset, model): pass
        def build_prompt(self, item, model):
            time.sleep(SLEEP_S)  # simulate RAG retrieval/transcription cost
            return "Is this unanswerable? Q: {question}"

    from src.mitigation.evaluation import evaluate_strategy
    result = evaluate_strategy(
        strategy=SlowBuildPromptStrategy(),
        dataset=dataset,
        model=mock_model,
        baseline_metrics={"f1": 0.5, "mcc": 0.0, "precision": 0.5,
                          "recall": 0.5, "specificity": 0.5, "balanced_accuracy": 0.5},
        model_id="test/model",
        corrupted_dataset_path=str(dataset_path),
    )

    assert result["records"][0]["inference_time_s"] >= SLEEP_S
