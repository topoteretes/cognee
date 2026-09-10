"""BROAD search type: answers "how many" questions by counting, not by reading (SDK-324).

One large prompt cannot answer a count: the model skims, and a count over a big
corpus needs every record, not the most similar ones. BROAD therefore never asks
an LLM for a number. It

1. plans: an LLM turns the question into a count spec — what one item is, where
   items live (typed graph entities or document text), an optional condition,
   an optional grouping, and what identifies a repeated mention of one item;
2. maps: every unit of the chosen source (entity records, or every document
   chunk / table row) is packed into shards, and parallel LLM calls LIST the
   matching items of each shard with a verbatim quote — listing is reliable,
   counting is not;
3. reduces in code: name variants are merged, repeated mentions of one item are
   removed, and the totals and per-group tallies are computed exactly;
4. answers: the final LLM only phrases the computed numbers.

Distinct things the graph already types (people, places, ...) are counted from
the graph directly, with name filters applied by code and no LLM pass; any other
condition is read from the text. How many times a word is written is counted by
code as well. The model only lists what needs reading.
"""

import asyncio
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.engine.utils import generate_node_name
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.shared.logging_utils import get_logger

logger = get_logger("BroadRetriever")

# Tokens of source text per map call. Small enough that the model lists every
# match in its shard; the corpus size only changes how many shards run. Measured:
# 2,000 CSV rows as text gave the top assignee 84/91 at 12k-token shards, 91/91 at 4k.
BROAD_SHARD_TOKENS = 4_000
BROAD_MAX_PARALLEL_CALLS = 16
# Graph node types whose text the text source reads: document chunks and table rows.
BROAD_TEXT_NODE_TYPES = ("DocumentChunk", "DltRow")
# Longest document first line (a CSV header, a title) repeated as shard context.
BROAD_PREAMBLE_CHARS = 400
# Entity types offered to the planner, most frequent first.
BROAD_MAX_PLANNER_TYPES = 200
# Name variants are merged in one LLM call; above this many names it is skipped.
BROAD_MAX_ALIAS_NAMES = 500
# Every group is shown so the answer can read one named group's tally.
BROAD_MAX_GROUPS_SHOWN = 500
BROAD_EVIDENCE_SHOWN = 8
_ENTITY_DESCRIPTION_CHARS = 200


class CountPlan(BaseModel):
    source: Literal["entities", "text"]
    entity_types: list[str] = []
    item: str
    name_contains: str | None = None
    literal_terms: list[str] = []
    condition: str | None = None
    group_by: str | None = None
    dedup_key: str | None = None
    dedup_key_is_name: bool = False


class ExtractedItem(BaseModel):
    unit: int
    key: str | None = None
    group: str | None = None
    evidence: str


class ShardItems(BaseModel):
    items: list[ExtractedItem]


class NameGroups(BaseModel):
    groups: list[list[str]]


@dataclass
class Unit:
    """One piece of source text the map step reads in full."""

    id: str
    text: str
    name: str = ""
    # First line of the unit's document (e.g. a CSV header) when the unit lacks it.
    preamble: str = ""


@dataclass
class CountResult:
    plan: CountPlan
    total: int
    groups: list[tuple[str, int]]
    units_scanned: int
    units_total: int
    evidence: list[str] = field(default_factory=list)
    names_merged: bool = True
    llm_calls: int = 0
    tokens_read: int = 0


PLAN_PROMPT = """You turn a question that asks for a count ("how many", "who has the \
most", "how often", "per ...") into a precise count specification. You never answer.

Choose the source:
- "entities" ONLY when the question asks how many distinct things of one kind exist \
or are mentioned, and that kind is one of the listed entity types — e.g. "How many \
cities appear?", "How many companies whose name starts with S are mentioned?". The \
graph holds each distinct entity once, so this count is exact. Entities carry only a \
name, so the only restriction allowed here is a name filter (name_contains); any other \
condition or a breakdown means source "text".
- "text" for everything else: what happened, who did what, relations between things, \
attributes or verdicts, how many times something occurs or is mentioned, per-person or \
per-group tallies of actions — e.g. "Which author wrote the most papers?", "How many \
orders were cancelled?", "How many times is Paris mentioned?". When unsure, choose "text".

Fields:
- entity_types: for source "entities", the exact type names from the list that the \
counted things belong to. Empty for "text".
- item: one countable item in plain words (e.g. "an order that was cancelled").
- name_contains: for source "entities", when the restriction is only that the name \
contains some text (e.g. "companies with Tech in their name" -> "Tech"), that text. It \
is matched exactly by code, so leave `condition` null in that case. Otherwise null.
- literal_terms: for source "text", ONLY when the question asks how many times a \
word or name is mentioned, appears or is used (e.g. "How many times is Paris \
mentioned?"), the exact spellings to match, with capitalization as written in the text \
(e.g. ["Paris"]). Code counts whole-word matches exactly. Empty for everything else — \
never for rows, records, events, or anything that needs reading to recognize.
- condition: a real restriction beyond the item itself that needs reading (e.g. \
"founded before 1900"), or null. Never "is mentioned", "exists" or "appears in the \
text" — that holds for everything.
- group_by: the attribute to tally items by when the question asks who/which has the \
most or asks for a breakdown, or null. When the question is about ONE named person or \
thing ("How many orders did Ann place?"), do not put that name in the condition: group \
by the attribute ("customer") instead. Names are written in several ways (nicknames, \
handles), and grouping merges the variants before the answer reads that one tally.
- dedup_key: when one item can be mentioned several times in the text (recaps, \
references back, lists repeated), the attribute that identifies it, with its format \
(e.g. "the PR number, digits only"). Null when every occurrence is its own item \
(counting mentions, headings, appearances). The key must identify ONE item across the \
whole corpus: if its values repeat for different items (chapter or section numbers that \
restart in every part, page numbers, "item 1"), use null. For "how many different X" it \
is X's name. Structural items — chapters, sections, pages, headings — are never \
repeated mentions: count each occurrence, dedup_key null.
- dedup_key_is_name: true when dedup_key is a name of a person/thing that may be \
written in different ways (nicknames, handles); false for numbers and codes."""

EXTRACT_PROMPT = """You extract items from a section of a larger corpus so they can \
be counted exactly by code. List EVERY matching item in this section, one entry per \
occurrence. Never count, estimate or summarize. Skip anything that does not satisfy \
the item definition and condition. For each entry give:
- key: the value of the identity attribute described below, in that exact format, \
or null if there is no identity attribute;
- group: the value of the grouping attribute described below, written as it appears \
(resolve pronouns like "they" to the name they refer to), or null if none;
- unit: the number N of the [unit N] block the item appears in;
- evidence: a short verbatim quote (at most 20 words) containing the item.
A "[document start — context only]" block repeats the first line of the document the \
following units come from (for example a table's column header). Use it to interpret \
those units; never list items from it.
Two items in different units are different items even when their quotes are identical \
(table rows often repeat the same values). Return an empty list if the section \
contains none."""

NAME_GROUPS_PROMPT = """You are given names that were extracted from one corpus. \
Group together the names that refer to the same person or thing (nicknames, handles \
with or without @, possessives, spelling variants). Only merge names you are sure \
about. Return every name exactly once; put the most complete form first in each \
group."""


def _normalize_key(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.lower())


class BroadRetriever(CompletionRetriever):
    """Count-and-aggregate search over every unit of a dataset."""

    def __init__(
        self,
        shard_tokens: int = BROAD_SHARD_TOKENS,
        max_parallel_calls: int = BROAD_MAX_PARALLEL_CALLS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shard_tokens = shard_tokens
        self.max_parallel_calls = max_parallel_calls
        self.tokenizer = TikTokenTokenizer()

    # --- sources ------------------------------------------------------------

    async def load_entities(self, graph_engine) -> dict[str, list[Unit]]:
        """Every typed entity in the graph, grouped by its normalized type name."""
        nodes, edges = await graph_engine.get_filtered_graph_data(
            [{"type": ["Entity", "EntityType"]}]
        )
        props_by_id = {str(node_id): props for node_id, props in nodes}
        entities_by_type: dict[str, list[Unit]] = {}
        for source_id, target_id, relationship_name, _ in edges:
            if relationship_name != "is_a":
                continue
            entity = props_by_id.get(str(source_id), {})
            entity_type = props_by_id.get(str(target_id), {})
            if entity.get("type") != "Entity" or entity_type.get("type") != "EntityType":
                continue
            description = (entity.get("description") or "")[:_ENTITY_DESCRIPTION_CHARS]
            entities_by_type.setdefault(entity_type["name"], []).append(
                Unit(
                    id=str(source_id),
                    text=f"{entity['name']}: {description}",
                    name=entity["name"],
                )
            )
        return entities_by_type

    async def load_text_units(self, graph_engine) -> list[Unit]:
        """Every document chunk and table row in the dataset, in document order.

        Listed from the graph, not found by vector search: a count needs every
        unit, and similarity plays no part in which units exist. A chunk from the
        middle of a document carries that document's first line: a CSV ingested
        as text has its column header only in chunk 0, and without it the model
        cannot tell one yes/no column from the next.
        """
        document_types = [cls.__name__ for cls in get_all_subclasses(Document)]
        nodes, edges = await graph_engine.get_filtered_graph_data(
            [{"type": [*BROAD_TEXT_NODE_TYPES, *document_types]}]
        )
        props_by_id = {str(node_id): props for node_id, props in nodes}
        document_of = {
            str(source): str(target)
            for source, target, relationship_name, _ in edges
            if relationship_name == "is_part_of"
        }
        chunks = {
            node_id: props
            for node_id, props in props_by_id.items()
            if props.get("type") in BROAD_TEXT_NODE_TYPES and props.get("text")
        }
        first_lines = {
            document_of[node_id]: props["text"].strip().split("\n", 1)[0][:BROAD_PREAMBLE_CHARS]
            for node_id, props in chunks.items()
            if node_id in document_of and str(props.get("chunk_index")) == "0"
        }
        units = []
        for node_id, props in sorted(
            chunks.items(),
            key=lambda item: (document_of.get(item[0], ""), int(item[1].get("chunk_index") or 0)),
        ):
            preamble = first_lines.get(document_of.get(node_id), "")
            if preamble in props["text"]:
                preamble = ""
            units.append(Unit(id=node_id, text=props["text"], preamble=preamble))
        logger.info(
            "BROAD units: %d, %d carrying their document's first line",
            len(units),
            sum(1 for unit in units if unit.preamble),
        )
        if not units:
            raise NoDataError("No data found in the system, please add data first.")
        return units

    # --- plan / map / reduce -------------------------------------------------

    async def plan(self, query: str, entities_by_type: dict[str, list[Unit]]) -> CountPlan:
        type_counts = sorted(
            ((name, len(units)) for name, units in entities_by_type.items()),
            key=lambda pair: -pair[1],
        )[:BROAD_MAX_PLANNER_TYPES]
        type_list = "\n".join(f"- {name} ({count} entities)" for name, count in type_counts)
        plan = await LLMGateway.acreate_structured_output(
            text_input=f"Question: {query}\n\nEntity types in the graph:\n{type_list or '(none)'}",
            system_prompt=PLAN_PROMPT,
            response_model=CountPlan,
        )
        if plan.source == "entities" and (plan.condition or plan.group_by):
            # A graph entity carries only a name and a short description, so a
            # condition or breakdown is read from the text, where the facts are.
            plan = plan.model_copy(
                update={"source": "text", "entity_types": [], "name_contains": None}
            )
        if plan.source == "entities":
            plan.entity_types = [generate_node_name(name) for name in plan.entity_types]
            unknown = [name for name in plan.entity_types if name not in entities_by_type]
            if unknown or not plan.entity_types:
                raise ValueError(f"BROAD planner chose entity types not in the graph: {unknown}")
        return plan

    def pack_shards(self, units: list[Unit], shard_tokens: int) -> tuple[list[list[Unit]], int]:
        """Group units into shards of at most ``shard_tokens``; returns (shards, tokens)."""
        shards: list[list[Unit]] = []
        total_tokens = 0
        current: list[Unit] = []
        current_tokens = 0
        for unit in units:
            tokens = len(self.tokenizer.extract_tokens(unit.text))
            if current and current_tokens + tokens > shard_tokens:
                shards.append(current)
                current, current_tokens = [], 0
            current.append(unit)
            current_tokens += tokens
            total_tokens += tokens
        if current:
            shards.append(current)
        return shards, total_tokens

    async def extract(self, plan: CountPlan, shard: list[Unit]) -> list[tuple[str, ExtractedItem]]:
        spec = (
            f"Item: {plan.item}\n"
            f"Condition: {plan.condition or 'none'}\n"
            f"Identity attribute (key): {plan.dedup_key or 'none'}\n"
            f"Grouping attribute (group): {plan.group_by or 'none'}"
        )
        # Unit markers let evidence be traced back; units are read in full. A
        # document's first line is shown once per shard, before its first unit.
        blocks: list[str] = []
        shown: set[str] = set()
        for index, unit in enumerate(shard):
            block = f"[unit {index}]\n{unit.text}"
            if unit.preamble and unit.preamble not in shown:
                shown.add(unit.preamble)
                block = f"[document start — context only]\n{unit.preamble}\n\n{block}"
            blocks.append(block)
        body = "\n\n".join(blocks)
        result = await LLMGateway.acreate_structured_output(
            text_input=f"{spec}\n\nSECTION:\n{body}",
            system_prompt=EXTRACT_PROMPT,
            response_model=ShardItems,
        )
        shard_id = shard[0].id
        return [(shard_id, item) for item in result.items]

    async def merge_names(self, names: set[str]) -> dict[str, str]:
        """Map each name variant to one canonical spelling."""
        if len(names) < 2:
            return {name: name for name in names}
        result = await LLMGateway.acreate_structured_output(
            text_input="\n".join(sorted(names)),
            system_prompt=NAME_GROUPS_PROMPT,
            response_model=NameGroups,
        )
        canonical = {name: name for name in names}
        for group in result.groups:
            for variant in group:
                if variant in canonical:
                    canonical[variant] = group[0]
        logger.info(
            "BROAD name merge: %d names, %d merged into another spelling",
            len(names),
            sum(1 for name, target in canonical.items() if name != target),
        )
        return canonical

    async def count_items(
        self, plan: CountPlan, units: list[Unit], shard_tokens: int
    ) -> CountResult:
        """Map every unit through the extractor, then dedupe, merge names and tally."""
        shards, tokens_read = self.pack_shards(units, shard_tokens)
        semaphore = asyncio.Semaphore(self.max_parallel_calls)

        async def run(shard):
            async with semaphore:
                return await self.extract(plan, shard)

        per_shard = await asyncio.gather(*(run(shard) for shard in shards))
        logger.info("BROAD map: %d units in %d shards", len(units), len(shards))

        # The model may list one occurrence twice; identical quotes from different
        # units (table rows repeating a value) are different items.
        seen: set[tuple[str, int, str]] = set()
        items: list[ExtractedItem] = []
        for shard_id, item in (pair for shard_items in per_shard for pair in shard_items):
            marker = (shard_id, item.unit, item.evidence.strip().lower())
            if marker not in seen:
                seen.add(marker)
                items.append(item)

        names_merged = True
        name_fields = []
        if plan.group_by:
            name_fields.append("group")
        if plan.dedup_key and plan.dedup_key_is_name:
            name_fields.append("key")
        names = {getattr(item, f) for item in items for f in name_fields if getattr(item, f)}
        if names and len(names) <= BROAD_MAX_ALIAS_NAMES:
            canonical = await self.merge_names(names)
            for item in items:
                for f in name_fields:
                    if getattr(item, f):
                        setattr(item, f, canonical[getattr(item, f)])
        elif names:
            names_merged = False

        if plan.dedup_key:
            unique: dict[str, ExtractedItem] = {}
            for item in items:
                key = _normalize_key(item.key) if item.key else None
                if key is None:
                    unique[f"__nokey_{len(unique)}"] = item
                elif key not in unique:
                    unique[key] = item
            items = list(unique.values())

        return CountResult(
            plan=plan,
            total=len(items),
            groups=Counter(item.group for item in items if item.group).most_common(),
            units_scanned=len(units),
            units_total=len(units),
            evidence=[item.evidence for item in items[:BROAD_EVIDENCE_SHOWN]],
            names_merged=names_merged,
            llm_calls=len(shards),
            tokens_read=tokens_read,
        )

    def count_literal(self, plan: CountPlan, units: list[Unit]) -> CountResult:
        """Whole-word, case-sensitive occurrences of the planned spellings in every unit."""
        terms = sorted(set(plan.literal_terms), key=len, reverse=True)
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, terms)) + r")(?!\w)")
        total = 0
        evidence: list[str] = []
        for unit in units:
            for match in pattern.finditer(unit.text):
                total += 1
                if len(evidence) < BROAD_EVIDENCE_SHOWN:
                    window = unit.text[max(match.start() - 60, 0) : match.end() + 60]
                    evidence.append(" ".join(window.split()))
        return CountResult(
            plan=plan,
            total=total,
            groups=[],
            units_scanned=len(units),
            units_total=len(units),
            evidence=evidence,
        )

    async def get_retrieved_objects(self, query: str) -> CountResult:
        unified_engine = await get_unified_engine()
        entities_by_type = await self.load_entities(unified_engine.graph)
        plan = await self.plan(query, entities_by_type)
        logger.info("BROAD plan: %s", plan.model_dump())

        if plan.source == "entities":
            all_units = [unit for name in plan.entity_types for unit in entities_by_type[name]]
            all_units = list({unit.id: unit for unit in all_units}.values())
            units = all_units
            if plan.name_contains:
                needle = generate_node_name(plan.name_contains)
                units = [unit for unit in all_units if needle in unit.name]
            # Graph entities are already distinct: the count is the node count.
            return CountResult(
                plan,
                len(units),
                [],
                len(all_units),
                len(all_units),
                [unit.text for unit in units[:BROAD_EVIDENCE_SHOWN]],
            )

        units = await self.load_text_units(unified_engine.graph)
        if plan.literal_terms and not plan.condition and not plan.group_by:
            # Occurrences of written words need no reading: code counts them exactly.
            return self.count_literal(plan, units)

        result = await self.count_items(plan, units, self.shard_tokens)
        logger.info(
            "BROAD result: total=%d calls=%d tokens=%d",
            result.total,
            result.llm_calls,
            result.tokens_read,
        )
        return result

    # --- context / completion --------------------------------------------------

    async def get_context_from_objects(self, query: str, retrieved_objects: CountResult) -> str:
        result = retrieved_objects
        plan = result.plan
        source = (
            f"graph entities of type {', '.join(plan.entity_types)}"
            if plan.source == "entities"
            else "document chunks / table rows"
        )
        how = (
            f"{result.llm_calls} parallel extraction calls over {result.tokens_read} tokens"
            if result.llm_calls
            else "no extraction calls needed"
        )
        lines = [
            (
                f"EXACT COUNT computed by code over all {result.units_scanned} of "
                f"{result.units_total} {source} (every unit read in full, no sampling; "
                f"{how}). Report these numbers; do not recount."
            ),
            f"Counted item: {plan.item}",
        ]
        if plan.source == "text" and plan.literal_terms and not result.llm_calls:
            lines.append(
                "Counted as whole-word matches of: "
                + ", ".join(f'"{term}"' for term in plan.literal_terms)
                + " (references by pronoun or description are not included)"
            )
        if plan.condition:
            lines.append(f"Condition: {plan.condition}")
        if plan.dedup_key:
            lines.append(f"Repeated mentions of one item removed by: {plan.dedup_key}")
        lines.append(f"TOTAL: {result.total}")
        if result.groups:
            lines.append(f"Tally by {plan.group_by} ({len(result.groups)} groups):")
            lines += [
                f"  {name}: {count}" for name, count in result.groups[:BROAD_MAX_GROUPS_SHOWN]
            ]
            if len(result.groups) > BROAD_MAX_GROUPS_SHOWN:
                lines.append(f"  ... {len(result.groups) - BROAD_MAX_GROUPS_SHOWN} more groups")
        if not result.names_merged:
            lines.append("Note: too many distinct names to merge spelling variants.")
        if result.evidence:
            lines.append("Examples:")
            lines += [f'  - "{quote}"' for quote in result.evidence]
        return "\n".join(lines)

    def extract_context_object_ids(self, retrieved_objects: Any) -> dict[str, list[str]] | None:
        return None

    def get_context_evidence(self, retrieved_objects: Any, dataset_id: Any = None):
        return None

    async def append_references(self, completions: list[Any], retrieved_objects: Any) -> list[Any]:
        return completions
