from .strategies.few_shot import FewShotStrategy
from .strategies.rag import RagStrategy

STRATEGIES: dict[str, type] = {
    "few_shot": FewShotStrategy,
    "rag": RagStrategy,
}
