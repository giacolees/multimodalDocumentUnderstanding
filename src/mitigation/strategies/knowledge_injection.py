"""Knowledge injection: prepend entity/layout/element metadata before querying the model."""

from dataclasses import dataclass, field


@dataclass
class DocumentMetadata:
    tables: list[str] = field(default_factory=list)      # e.g. ["Table 1", "Table 2"]
    figures: list[str] = field(default_factory=list)
    layout_regions: list[str] = field(default_factory=list)  # e.g. ["header", "footer"]
    entities: dict[str, list[str]] = field(default_factory=dict)  # type → values


def build_knowledge_injection_prompt(
    question: str,
    metadata: DocumentMetadata,
) -> str:
    lines = ["Document metadata (use this to decide answerability):"]
    if metadata.tables:
        lines.append(f"  Tables present: {', '.join(metadata.tables)}")
    if metadata.figures:
        lines.append(f"  Figures present: {', '.join(metadata.figures)}")
    if metadata.layout_regions:
        lines.append(f"  Layout regions: {', '.join(metadata.layout_regions)}")
    for etype, vals in metadata.entities.items():
        lines.append(f"  {etype}: {', '.join(vals)}")
    lines.append("")
    lines.append(
        "Given the document image and the metadata above, answer UNANSWERABLE if the question "
        "cannot be answered, otherwise provide the answer."
    )
    lines.append(f"\nQuestion: {question}")
    return "\n".join(lines)


from .base import MitigationStrategy


class KnowledgeInjectionStrategy(MitigationStrategy):
    name = "knowledge_injection"

    def __init__(self, config: dict) -> None:
        pass

    def build_prompt(self, item: dict, model) -> str:
        return build_knowledge_injection_prompt(item["corrupted_question"], DocumentMetadata())
