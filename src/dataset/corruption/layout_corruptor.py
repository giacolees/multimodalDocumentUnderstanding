import re
from typing import Optional
from .base_corruptor import BaseCorruptor, CorruptedSample, CorruptionType


# Sorted longest-first so the regex prefers the most specific match
_LAYOUT_PHRASES = sorted([
    # corner — explicit
    "top left corner", "top right corner", "bottom left corner", "bottom right corner",
    "upper left corner", "upper right corner", "lower left corner", "lower right corner",
    # two-word position
    "top left", "top right", "bottom left", "bottom right",
    "upper left", "upper right", "lower left", "lower right",
    "top center", "bottom center",
    # structural
    "left margin", "right margin",
    "left side", "right side",
    "on the left", "on the right",
    "header", "footer",
    # full-page anchors
    "top of the page", "bottom of the page",
    "top of the document", "bottom of the document",
    "center of the page", "middle of the page",
    # column references
    "first column", "second column", "last column",
], key=len, reverse=True)

_LAYOUT_RE = re.compile(
    "|".join(re.escape(p) for p in _LAYOUT_PHRASES), re.IGNORECASE
)

# Maps each phrase to spatially opposite replacements so the judge reliably
# marks the corrupted question as unanswerable (opposite region ≠ original region).
_OPPOSITES: dict[str, list[str]] = {
    "top left corner":      ["bottom right corner", "bottom left corner", "bottom right"],
    "top right corner":     ["bottom left corner",  "bottom right corner", "bottom left"],
    "bottom left corner":   ["top right corner",    "top left corner",    "top right"],
    "bottom right corner":  ["top left corner",     "top right corner",   "top left"],
    "upper left corner":    ["lower right corner",  "lower left corner",  "bottom right"],
    "upper right corner":   ["lower left corner",   "lower right corner", "bottom left"],
    "lower left corner":    ["upper right corner",  "upper left corner",  "top right"],
    "lower right corner":   ["upper left corner",   "upper right corner", "top left"],
    "top left":             ["bottom right", "bottom left",  "bottom center"],
    "top right":            ["bottom left",  "bottom right", "bottom center"],
    "bottom left":          ["top right",    "top left",     "top center"],
    "bottom right":         ["top left",     "top right",    "top center"],
    "upper left":           ["lower right",  "lower left",   "bottom right"],
    "upper right":          ["lower left",   "lower right",  "bottom left"],
    "lower left":           ["upper right",  "upper left",   "top right"],
    "lower right":          ["upper left",   "upper right",  "top left"],
    "top center":           ["bottom center", "bottom left", "bottom right"],
    "bottom center":        ["top center",    "top left",    "top right"],
    "left margin":          ["right margin"],
    "right margin":         ["left margin"],
    "left side":            ["right side"],
    "right side":           ["left side"],
    "on the left":          ["on the right"],
    "on the right":         ["on the left"],
    "header":               ["footer", "bottom of the page"],
    "footer":               ["header", "top of the page"],
    "top of the page":      ["bottom of the page", "footer"],
    "bottom of the page":   ["top of the page",    "header"],
    "top of the document":  ["bottom of the document"],
    "bottom of the document": ["top of the document"],
    "center of the page":   ["top left corner", "bottom right corner"],
    "middle of the page":   ["top left",        "bottom right"],
    "first column":         ["last column"],
    "last column":          ["first column"],
    "second column":        ["first column", "last column"],
}


class LayoutCorruptor(BaseCorruptor):
    """Replaces layout position references with spatially opposite alternatives."""

    corruption_type = CorruptionType.LAYOUT

    def corrupt(self, question: str, context: Optional[dict] = None) -> Optional[CorruptedSample]:
        m = _LAYOUT_RE.search(question)
        if not m:
            return None

        original = m.group()
        # Prefer opposite-region replacements; fall back to any different phrase
        candidates = _OPPOSITES.get(original.lower(), [])
        candidates = [c for c in candidates if c.lower() != original.lower()]
        if not candidates:
            candidates = [p for p in _LAYOUT_PHRASES if p.lower() != original.lower()]

        replacement = self.rng.choice(candidates)
        corrupted = question[: m.start()] + replacement + question[m.end() :]
        return CorruptedSample(
            original_question=question,
            corrupted_question=corrupted,
            corruption_type=self.corruption_type,
            corruption_detail=f"layout:'{original}'→'{replacement}'",
        )
