import re
from typing import Optional
from .base_corruptor import BaseCorruptor, CorruptedSample, CorruptionType


_LAYOUT_PHRASES = [
    "top left", "top right", "bottom left", "bottom right",
    "top center", "bottom center", "left margin", "right margin",
    "header", "footer", "top of the page", "bottom of the page",
    "first column", "second column", "last column",
]

_LAYOUT_RE = re.compile(
    "|".join(re.escape(p) for p in _LAYOUT_PHRASES), re.IGNORECASE
)


class LayoutCorruptor(BaseCorruptor):
    """Replaces layout position references with incorrect alternatives."""

    corruption_type = CorruptionType.LAYOUT

    def corrupt(self, question: str, context: Optional[dict] = None) -> Optional[CorruptedSample]:
        m = _LAYOUT_RE.search(question)
        if not m:
            return None

        original = m.group()
        alternatives = [p for p in _LAYOUT_PHRASES if p.lower() != original.lower()]
        replacement = self.rng.choice(alternatives)
        corrupted = question[: m.start()] + replacement + question[m.end() :]
        return CorruptedSample(
            original_question=question,
            corrupted_question=corrupted,
            corruption_type=self.corruption_type,
            corruption_detail=f"layout:'{original}'→'{replacement}'",
        )
