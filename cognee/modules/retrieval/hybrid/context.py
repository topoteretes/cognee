from dataclasses import dataclass, field
from typing import Any, Optional

from cognee.modules.retrieval.hybrid.entities import format_entities
from cognee.modules.retrieval.hybrid.facts import format_facts
from cognee.modules.retrieval.hybrid.results import display_value, payload, result_id


@dataclass
class _ContextSection:
    heading: Optional[str]
    separator: str
    items: list[str] = field(default_factory=list)
    item_ids: list[Optional[str]] = field(default_factory=list)


def format_hybrid_context(
    global_context: str,
    retrieved_objects: Any,
    *,
    max_context_chars: Optional[int] = None,
    max_context_items: Optional[int] = None,
    selected_chunk_ids: Optional[set[str]] = None,
) -> str:
    """Build bounded hybrid context from whole evidence items.

    Items are considered in a fixed order: passages, graph fallback, entities,
    facts, then optional global context. An item that does not fit is omitted rather
    than sliced, so references, graph statements, and source passages are never
    cut into misleading fragments.
    """
    retrieved_objects = retrieved_objects or {}
    sections = _context_sections(global_context, retrieved_objects)
    return _select_bounded_items(
        sections,
        max_context_chars,
        max_context_items,
        selected_chunk_ids=selected_chunk_ids,
    )


def _context_sections(global_context: str, retrieved_objects: dict) -> list[_ContextSection]:
    sections = []

    passage_items = _passage_items_with_ids(
        retrieved_objects.get("chunks", []),
        retrieved_objects.get("chunk_summaries", {}),
    )
    if passage_items:
        sections.append(
            _ContextSection(
                "## Relevant passages",
                "\n---\n",
                [item[0] for item in passage_items],
                [item[1] for item in passage_items],
            )
        )

    graph_fallback_context = display_value(retrieved_objects.get("graph_fallback_context"))
    if graph_fallback_context:
        sections.append(
            _ContextSection(
                "## Graph fallback evidence",
                "\n",
                [graph_fallback_context],
            )
        )

    entity_items = _entity_items(retrieved_objects.get("entities", []))
    if entity_items:
        sections.append(_ContextSection("## Relevant entities", "\n\n", entity_items))

    fact_items = _fact_items(retrieved_objects.get("facts", []))
    if fact_items:
        sections.append(_ContextSection("## Related facts", "\n", fact_items))

    if global_context:
        sections.append(_ContextSection(None, "\n", [global_context]))

    return sections


def _select_bounded_items(
    sections: list[_ContextSection],
    max_context_chars: Optional[int],
    max_context_items: Optional[int],
    *,
    selected_chunk_ids: Optional[set[str]] = None,
) -> str:
    if max_context_chars is not None and max_context_chars <= 0:
        return ""
    if max_context_items is not None and max_context_items <= 0:
        return ""

    selected = [_ContextSection(section.heading, section.separator) for section in sections]
    selected_items = 0
    for section_index, section in enumerate(sections):
        for item_index, item in enumerate(section.items):
            if max_context_items is not None and selected_items >= max_context_items:
                break

            selected[section_index].items.append(item)
            candidate = _render_sections(selected)
            if max_context_chars is not None and len(candidate) > max_context_chars:
                selected[section_index].items.pop()
                continue
            item_id = section.item_ids[item_index] if item_index < len(section.item_ids) else None
            if item_id and selected_chunk_ids is not None:
                selected_chunk_ids.add(item_id)
            selected_items += 1

    return _render_sections(selected)


def _render_sections(sections: list[_ContextSection]) -> str:
    blocks = []
    for section in sections:
        if not section.items:
            continue
        body = section.separator.join(section.items)
        blocks.append(f"{section.heading}\n{body}" if section.heading else body)
    return "\n\n".join(blocks)


def extract_context_object_ids(retrieved_objects: Any) -> Optional[dict[str, list[str]]]:
    # Facts are intentionally excluded: their ids are EdgeType vector rows, not graph nodes.
    if not isinstance(retrieved_objects, dict):
        return None

    node_ids = set()
    for chunk in retrieved_objects.get("chunks", []):
        chunk_id = result_id(chunk)
        if chunk_id:
            node_ids.add(chunk_id)

    for entity in retrieved_objects.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_id = display_value(entity.get("id"))
        if entity_id:
            node_ids.add(entity_id)
        for edge in entity.get("edges", []):
            if not isinstance(edge, dict):
                continue
            for key in ("source_id", "target_id"):
                edge_node_id = display_value(edge.get(key))
                if edge_node_id:
                    node_ids.add(edge_node_id)

    return {"node_ids": sorted(node_ids)} if node_ids else None


def format_passages(chunks: list[Any], chunk_summaries: Optional[dict[str, str]] = None) -> str:
    texts = _passage_items(chunks, chunk_summaries)
    if not texts:
        return ""
    return "## Relevant passages\n" + "\n---\n".join(texts)


def _passage_items(
    chunks: list[Any], chunk_summaries: Optional[dict[str, str]] = None
) -> list[str]:
    return [item[0] for item in _passage_items_with_ids(chunks, chunk_summaries)]


def _passage_items_with_ids(
    chunks: list[Any], chunk_summaries: Optional[dict[str, str]] = None
) -> list[tuple[str, Optional[str]]]:
    texts = []
    chunk_summaries = chunk_summaries or {}
    for chunk in chunks or []:
        text = display_value(payload(chunk).get("text"))
        if not text:
            continue

        chunk_id = result_id(chunk)
        summary_text = chunk_summaries.get(chunk_id) if chunk_id else None
        if summary_text:
            texts.append((f"[Passage Summary]: {summary_text}\n[Raw Passage]: {text}", chunk_id))
        else:
            texts.append((text, chunk_id))
    return texts


def _entity_items(entities: list[dict]) -> list[str]:
    items = []
    prefix = "## Relevant entities\n"
    for entity in entities or []:
        block = format_entities([entity])
        if block.startswith(prefix):
            items.append(block[len(prefix) :])
    return items


def _fact_items(facts: list[dict]) -> list[str]:
    items = []
    prefix = "## Related facts\n"
    for fact in facts or []:
        block = format_facts([fact])
        if block.startswith(prefix):
            items.append(block[len(prefix) :])
    return items
