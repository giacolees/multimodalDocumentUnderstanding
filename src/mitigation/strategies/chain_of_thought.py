"""Chain-of-thought prompt builder for unanswerable question detection."""


_COT_TEMPLATE = """Look at the document image carefully.

Think step by step:
1. What elements are visible in the document (tables, figures, charts, text sections, layout regions)?
2. What does the question ask for?
3. Is the specific element or information the question refers to actually present in the document?
4. Based on your analysis, is the question ANSWERABLE or UNANSWERABLE?

Provide your reasoning first, then conclude with either ANSWERABLE or UNANSWERABLE.

Question: {question}

Reasoning:"""


def build_cot_prompt(question: str) -> str:
    return _COT_TEMPLATE.format(question=question)


from .base import MitigationStrategy, get_question


class ChainOfThoughtStrategy(MitigationStrategy):
    name = "chain_of_thought"

    def __init__(self, config: dict) -> None:
        pass

    def build_prompt(self, item: dict, model) -> str:
        return build_cot_prompt(get_question(item))
