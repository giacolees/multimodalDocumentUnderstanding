"""NLP entity corruptor backed by spaCy NER + Wikipedia category lookup.

For each named entity detected by spaCy the corruptor:
  1. Queries Wikipedia for the entity's article categories.
  2. Picks a category likely to contain peer entities (same type/domain).
  3. Fetches category members and uses them as replacement candidates.

This avoids hardcoded pools: if spaCy finds "Toyota" (ORG) the replacement
comes from actual sibling companies in Wikipedia, not a generic list.

Temporal entities (DATE / TIME) are handled locally since Wikipedia
categories do not help there.

Falls back to a small static pool when:
  - Wikipedia has no article for the entity.
  - The network call fails.
  - No usable category is found.

Constructor args:
  spacy_model    – spaCy model to load (default: en_core_web_sm).
                   Install: uv run python -m spacy download en_core_web_sm
  web_lookup     – set False to skip Wikipedia and always use the fallback pools.
  wiki_timeout   – seconds per Wikipedia API call (default: 8).
  max_candidates – maximum number of candidates fetched per category (default: 40).
"""

import re
from typing import Optional

import requests  # type: ignore[import-untyped]

from .base_corruptor import BaseCorruptor, CorruptedSample, CorruptionType

# ---------------------------------------------------------------------------
# Temporal helpers (Wikipedia is not useful here)
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.IGNORECASE)
_YEARS_POOL = list(range(1950, 2025))

# ---------------------------------------------------------------------------
# Static fallback pools (used only when Wikipedia lookup fails)
# ---------------------------------------------------------------------------

_FALLBACK: dict[str, list[str]] = {
    "GPE": [
        "France", "Germany", "Italy", "Spain", "Brazil", "Canada", "Japan",
        "Australia", "India", "Mexico", "South Korea", "Netherlands", "Sweden",
    ],
    "LOC": [
        "the Pacific Ocean", "the Amazon basin", "the Alps", "the Sahara Desert",
        "the Great Barrier Reef", "the Mississippi River",
    ],
    "ORG": [
        "Acme Corp", "GlobalTech", "Meridian Industries", "NovaSystems",
        "Crestwood Group", "Pinnacle Solutions", "Apex Dynamics",
    ],
    "PERSON": [
        "John Smith", "Maria Garcia", "Wei Zhang", "Priya Patel", "Luca Rossi",
        "Emma Müller", "Carlos López", "Yuki Tanaka",
    ],
}

# ---------------------------------------------------------------------------
# Wikipedia category helpers
# ---------------------------------------------------------------------------

_WIKI_API = "https://en.wikipedia.org/w/api.php"

# Category title fragments that tend to contain peer entities of the right type.
# Checked in order; first matching category is used.
_CATEGORY_PREFERENCES: dict[str, list[str]] = {
    "GPE": ["countries", "cities", "states", "nations", "municipalities"],
    "LOC": ["regions", "mountains", "rivers", "lakes", "oceans", "geographic"],
    "ORG": ["companies", "corporations", "organisations", "organizations",
            "airlines", "banks", "universities", "clubs", "agencies"],
    "PERSON": ["people", "politicians", "scientists", "artists", "athletes",
               "writers", "leaders", "directors", "actors"],
}


def _wiki_categories(title: str, timeout: int) -> list[str]:
    """Return visible Wikipedia categories for *title*."""
    try:
        resp = requests.get(
            _WIKI_API,
            params={
                "action": "query", "format": "json",
                "titles": title, "prop": "categories",
                "cllimit": 30, "clshow": "!hidden",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            return []
        return [c["title"] for c in page.get("categories", [])]
    except Exception:
        return []


def _wiki_category_members(category: str, limit: int, timeout: int) -> list[str]:
    """Return page titles that are members of *category*."""
    try:
        resp = requests.get(
            _WIKI_API,
            params={
                "action": "query", "format": "json",
                "list": "categorymembers", "cmtitle": category,
                "cmlimit": limit, "cmtype": "page",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return [m["title"] for m in resp.json()["query"]["categorymembers"]]
    except Exception:
        return []


def _best_category(categories: list[str], label: str) -> Optional[str]:
    """Pick the most relevant category for the entity label."""
    prefs = _CATEGORY_PREFERENCES.get(label, [])
    for cat in categories:
        lower = cat.lower()
        if any(kw in lower for kw in prefs):
            return cat
    return None


# ---------------------------------------------------------------------------
# Main corruptor
# ---------------------------------------------------------------------------


class NLPEntityCorruptor(BaseCorruptor):
    """Replaces named entities with Wikipedia-sourced peer entities."""

    corruption_type = CorruptionType.NLP_ENTITY

    def __init__(
        self,
        seed: int = 42,
        spacy_model: str = "en_core_web_sm",
        web_lookup: bool = True,
        wiki_timeout: int = 8,
        max_candidates: int = 40,
    ) -> None:
        super().__init__(seed)
        try:
            import spacy as _spacy  # type: ignore[import-untyped]
            self._nlp = _spacy.load(spacy_model)
        except OSError:
            raise ImportError(
                f"spaCy model '{spacy_model}' not found. "
                f"Run: uv run python -m spacy download {spacy_model}"
            )
        self._web_lookup = web_lookup
        self._wiki_timeout = wiki_timeout
        self._max_candidates = max_candidates
        # {"{label}:{entity_text}": [candidate, ...]}
        self._cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------

    def corrupt(self, question: str, context: Optional[dict] = None) -> Optional[CorruptedSample]:
        doc = self._nlp(question)
        for ent in doc.ents:
            result = self._corrupt_entity(ent, question)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------

    def _corrupt_entity(self, ent, question: str) -> Optional[CorruptedSample]:
        label = ent.label_
        span = ent.text
        start, end = ent.start_char, ent.end_char

        if label in ("DATE", "TIME"):
            return self._corrupt_temporal(span, question, start, end)

        candidates = self._get_candidates(span, label)
        candidates = [c for c in candidates if c.lower() != span.lower()]
        if not candidates:
            return None

        replacement = self.rng.choice(candidates)
        return self._make_sample(question, span, replacement, start, end, label)

    def _corrupt_temporal(
        self, span: str, question: str, start: int, end: int
    ) -> Optional[CorruptedSample]:
        m = _YEAR_RE.search(span)
        if m:
            original_year = int(m.group())
            pool = [y for y in _YEARS_POOL if abs(y - original_year) > 5]
            new_span = span[: m.start()] + str(self.rng.choice(pool)) + span[m.end() :]
            return self._make_sample(question, span, new_span, start, end, "DATE/year")

        m = _MONTH_RE.search(span)
        if m:
            original = m.group()
            candidates = [mo for mo in _MONTHS if mo.lower() != original.lower()]
            new_span = span[: m.start()] + self.rng.choice(candidates) + span[m.end() :]
            return self._make_sample(question, span, new_span, start, end, "DATE/month")

        return None

    # ------------------------------------------------------------------

    def _get_candidates(self, entity_text: str, label: str) -> list[str]:
        cache_key = f"{label}:{entity_text.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        candidates: list[str] = []

        if self._web_lookup:
            candidates = self._wikipedia_lookup(entity_text, label)

        if not candidates:
            candidates = list(_FALLBACK.get(label, []))

        self._cache[cache_key] = candidates
        return candidates

    def _wikipedia_lookup(self, entity_text: str, label: str) -> list[str]:
        categories = _wiki_categories(entity_text, self._wiki_timeout)
        if not categories:
            return []

        category = _best_category(categories, label)
        if not category:
            return []

        members = _wiki_category_members(category, self._max_candidates, self._wiki_timeout)
        # Strip disambiguation suffixes like "France (country)"
        return [m.split(" (")[0] for m in members if m != entity_text]

    # ------------------------------------------------------------------

    @staticmethod
    def _make_sample(
        question: str, original: str, replacement: str, start: int, end: int, label: str
    ) -> CorruptedSample:
        return CorruptedSample(
            original_question=question,
            corrupted_question=question[:start] + replacement + question[end:],
            corruption_type=CorruptionType.NLP_ENTITY,
            corruption_detail=f"{label}:{original}→{replacement}",
        )
