import re
from typing import Optional
from .base_corruptor import BaseCorruptor, CorruptedSample, CorruptionType


_ELEMENTS = ["Table", "Figure", "Chart", "Graph", "Footnote", "Appendix", "Section", "Exhibit"]
# Number is optional — "the Table" is as valid a target as "Table 3"
_ELEMENT_RE = re.compile(
    r"\b(" + "|".join(_ELEMENTS) + r")(?:\s+(\d+))?", re.IGNORECASE
)


class ElementCorruptor(BaseCorruptor):
    """Replaces document element references (Table 1 → Figure 2, bare Table → Chart, etc.)."""

    corruption_type = CorruptionType.ELEMENT

    def corrupt(self, question: str, context: Optional[dict] = None) -> Optional[CorruptedSample]:
        m = _ELEMENT_RE.search(question)
        if not m:
            return None

        original_element = m.group(1)
        original_number = m.group(2)  # may be None

        alt_elements = [e for e in _ELEMENTS if e.lower() != original_element.lower()]
        new_element = self.rng.choice(alt_elements)

        if original_number is not None:
            # Change element type and shift the number so both differ
            orig_num = int(original_number)
            delta = self.rng.choice([-2, -1, 1, 2, 3])
            new_num = max(1, orig_num + delta)
            replacement = f"{new_element} {new_num}"
            detail = f"element:{original_element} {original_number}→{replacement}"
        else:
            # Bare reference — just swap the element type
            replacement = new_element
            detail = f"element:{original_element}→{replacement}"

        corrupted = question[: m.start()] + replacement + question[m.end() :]
        return CorruptedSample(
            original_question=question,
            corrupted_question=corrupted,
            corruption_type=self.corruption_type,
            corruption_detail=detail,
        )
