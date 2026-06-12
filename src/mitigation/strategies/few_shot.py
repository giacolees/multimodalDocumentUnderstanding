"""Few-shot prompt builder: injects k examples before the target question."""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class FewShotExample:
    question: str
    is_unanswerable: bool
    explanation: str


_DEFAULT_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        question="What is the value in Table 3?",
        is_unanswerable=False,
        explanation="Table 3 is visible and contains values.",
    ),
    FewShotExample(
        question="What is the Net Requirement in Table 5?",
        is_unanswerable=True,
        explanation="There is no Table 5 in this document.",
    ),
    FewShotExample(
        question="What year is shown in the bottom right chart?",
        is_unanswerable=False,
        explanation="A year is visible in the bottom right chart.",
    ),
    FewShotExample(
        question="What figure is located at the top left?",
        is_unanswerable=True,
        explanation="There is no figure at the top left of this page.",
    ),
]


def build_few_shot_prompt(
    question: str,
    examples: Sequence[FewShotExample] = _DEFAULT_EXAMPLES,
    k: int = 2,
) -> str:
    header = (
        "You will be shown a document image and a question. "
        "Decide if the question is ANSWERABLE or UNANSWERABLE based solely on the document.\n\n"
        "Examples:\n"
    )
    example_block = ""
    for ex in list(examples)[:k]:
        label = "UNANSWERABLE" if ex.is_unanswerable else "ANSWERABLE"
        example_block += f"Q: {ex.question}\nA: {label} — {ex.explanation}\n\n"
    return header + example_block + f"Now answer:\nQ: {question}\nA:"


from .base import MitigationStrategy


class FewShotStrategy(MitigationStrategy):
    name = "few_shot"

    def __init__(self, config: dict) -> None:
        self._k = config.get("k", 2)

    def build_prompt(self, item: dict, model) -> str:
        return build_few_shot_prompt(item["corrupted_question"], k=self._k)
