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
