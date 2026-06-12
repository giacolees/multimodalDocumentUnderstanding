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
