from .strategies.few_shot import FewShotStrategy
from .strategies.chain_of_thought import ChainOfThoughtStrategy
from .strategies.knowledge_injection import KnowledgeInjectionStrategy
from .strategies.rag import RagStrategy

STRATEGIES: dict[str, type] = {
    "few_shot": FewShotStrategy,
    "chain_of_thought": ChainOfThoughtStrategy,
    "knowledge_injection": KnowledgeInjectionStrategy,
    "rag": RagStrategy,
}
