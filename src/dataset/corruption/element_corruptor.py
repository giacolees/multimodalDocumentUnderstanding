import re
from typing import Optional
from .base_corruptor import BaseCorruptor, CorruptedSample, CorruptionType


_ELEMENTS = ["Table", "Figure", "Chart", "Graph", "Footnote", "Appendix", "Section", "Exhibit"]
_ELEMENT_RE = re.compile(
    r"\b(" + "|".join(_ELEMENTS) + r")\s*(\d+)", re.IGNORECASE
)


class ElementCorruptor(BaseCorruptor):
    """Replaces document element references (Table 1 → Figure 2, etc.)."""

    corruption_type = CorruptionType.ELEMENT

    def corrupt(self, question: str, context: Optional[dict] = None) -> Optional[CorruptedSample]:
        m = _ELEMENT_RE.search(question)
        if not m:
            return None

        original_element = m.group(1)
        original_number = int(m.group(2))

        alt_elements = [e for e in _ELEMENTS if e.lower() != original_element.lower()]
        new_element = self.rng.choice(alt_elements)
        # Use a different number (±1-3, at least 1)
        delta = self.rng.choice([-2, -1, 1, 2, 3])
        new_number = max(1, original_number + delta)

        replacement = f"{new_element} {new_number}"
        corrupted = question[: m.start()] + replacement + question[m.end() :]
        return CorruptedSample(
            original_question=question,
            corrupted_question=corrupted,
            corruption_type=self.corruption_type,
            corruption_detail=f"element:{original_element} {original_number}→{replacement}",
        )
