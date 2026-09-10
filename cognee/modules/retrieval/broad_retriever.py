"""BROAD search type: answers "how many" questions by counting in code (SDK-324).

A model cannot count a large corpus by reading it: it skims, and a count needs
every record, not the most similar ones. BROAD never asks a model for a number.
One LLM call plans the count, then one of three counters runs:

- graph:   distinct things of a type the graph holds ("how many people") are its
           entity nodes, optionally filtered by name. Exact, no LLM calls.
- words:   how many times a word is written is a whole-word match over every
           chunk and table row. Exact for the planned spellings, no LLM calls.
- reading: everything else (events, relations, verdicts). Every chunk and row is
           read in parallel shards; each call LISTS the matching items with a
           quote, and code merges name variants, drops repeated mentions and
           tallies. The tally is exact over what was listed; an item the reading
           missed is not in it, and the answer says so.

The final LLM call only phrases the computed numbers.
"""

import asyncio
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.engine.utils import generate_node_name
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.shared.logging_utils import get_logger

logger = get_logger("BroadRetriever")

# Tokens of source text per reading call. Small enough that the model lists every
# match in its shard; the corpus size only changes how many shards run. Measured:
# 2,000 CSV rows as text gave the top assignee 84/91 at 12k-token shards, 91/91 at 4k.
BROAD_SHARD_TOKENS = 4_000
BROAD_MAX_PARALLEL_CALLS = 16
# Graph node types whose text is read: document chunks and table rows.
BROAD_TEXT_NODE_TYPES = ("DocumentChunk", "DltRow")
# Longest document first line (a CSV header, a title) repeated as shard context.
BROAD_PREAMBLE_CHARS = 400
# Entity types offered to the planner, most frequent first.
BROAD_MAX_PLANNER_TYPES = 200
# Name variants are merged in one LLM call; above this many names it is skipped.
BROAD_MAX_ALIAS_NAMES = 500
# Every group is shown so the answer can read one named group's tally.
BROAD_MAX_GROUPS_SHOWN = 500
# Listed items quoted in the answer context: all of them behind a small count, so
# the answer can cite them or, for a question that is not a count, answer from them.
BROAD_EVIDENCE_SHOWN = 50
# A question about one named person or thing lists that target's items in full.
BROAD_TARGET_ITEMS_SHOWN = 1_000
# Longest list appended to an answer that asks to list the counted items.
BROAD_MAX_LISTED = 10_000
_ENTITY_DESCRIPTION_CHARS = 200


class CountPlan(BaseModel):
    source: Literal["entities", "text"]
    entity_types: list[str] = []
    item: str
    name_contains: str | None = None
    literal_terms: list[str] = []
    condition: str | None = None
    group_by: str | None = None
    # "How many different X": the answer is the number of groups, not of items.
    distinct: bool = False
    # "How many orders did Ann place": the one group_by value asked about, as written.
    target: str | None = None
    # "How many units were shipped": the numeric attribute summed instead of counting items.
    measure: str | None = None
    # "... and list them": the full list of counted items is appended to the answer by code.
    list_items: bool = False
    dedup_key: str | None = None


class ExtractedItem(BaseModel):
    unit: int
    key: str | None = None
    group: str | None = None
    # The item's value of the plan's measure, when the plan sums one.
    amount: float | None = None
    evidence: str


class ShardItems(BaseModel):
    items: list[ExtractedItem]
    # Names the text itself says are one person or thing ("Akshats-git, usually called Akshats").
    aliases: list[list[str]] = []


class NameGroups(BaseModel):
    groups: list[list[str]]


class TargetMatch(BaseModel):
    names: list[str]


@dataclass
class Unit:
    """One piece of source text: an entity record, a document chunk or a table row."""

    id: str
    text: str
    name: str = ""
    # First line of the unit's document (e.g. a CSV header) when the unit lacks it.
    preamble: str = ""


@dataclass
class CountResult:
    plan: CountPlan
    method: Literal["graph", "words", "reading"]
    total: float
    units: int
    groups: list[tuple[str, float]] = field(default_factory=list)
    # Every counted item as one line; the answer context shows a sample, a listing shows all.
    evidence: list[str] = field(default_factory=list)
    # For a plan with a measure: listed items that stated no amount (added as 0).
    amounts_missing: int = 0
    # For a plan with a target: the corpus names matched to it; empty when none matched.
    target_names: list[str] = field(default_factory=list)
    names_merged: bool = True
    items_listed: int = 0
    llm_calls: int = 0
    tokens_read: int = 0


def _read_prompt(name: str) -> str:
    prompt = read_query_prompt(name)
    if prompt is None:
        raise FileNotFoundError(f"BROAD prompt {name!r} could not be read.")
    return prompt


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:,.2f}"


def _stated_aliases(aliases: list[list[str]]) -> list[str]:
    return sorted({" = ".join(group) for group in aliases if len(group) > 1})


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

    async def get_retrieved_objects(self, query: str) -> CountResult:
        graph_engine = (await get_unified_engine()).graph
        entities_by_type = await self.load_entities(graph_engine)
        plan = await self.plan(query, entities_by_type)
        logger.info("BROAD plan: %s", plan.model_dump())

        if plan.source == "entities":
            result = self.count_entities(plan, entities_by_type)
        else:
            units = await self.load_text_units(graph_engine)
            if plan.literal_terms and not (plan.condition or plan.group_by or plan.measure):
                result = self.count_words(plan, units)
            else:
                result = await self.count_by_reading(plan, units)

        logger.info(
            "BROAD %s count: total=%d over %d units, %d LLM calls, %d tokens read",
            result.method,
            result.total,
            result.units,
            result.llm_calls,
            result.tokens_read,
        )
        return result

    # --- sources --------------------------------------------------------------

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
        document_of = {
            str(source): str(target)
            for source, target, relationship_name, _ in edges
            if relationship_name == "is_part_of"
        }
        chunks = {
            str(node_id): props
            for node_id, props in nodes
            if props.get("type") in BROAD_TEXT_NODE_TYPES and props.get("text")
        }
        if not chunks:
            raise NoDataError("No data found in the system, please add data first.")

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
        return units

    # --- planning ---------------------------------------------------------------

    async def plan(self, query: str, entities_by_type: dict[str, list[Unit]]) -> CountPlan:
        type_counts = sorted(
            ((name, len(units)) for name, units in entities_by_type.items()),
            key=lambda pair: -pair[1],
        )[:BROAD_MAX_PLANNER_TYPES]
        type_list = "\n".join(f"- {name} ({count} entities)" for name, count in type_counts)
        plan = await LLMGateway.acreate_structured_output(
            text_input=f"Question: {query}\n\nEntity types in the graph:\n{type_list or '(none)'}",
            system_prompt=_read_prompt("broad_plan.txt"),
            response_model=CountPlan,
        )
        if plan.target and not plan.group_by:
            raise ValueError(
                f"BROAD planner named a target ({plan.target!r}) without the attribute it is a "
                "value of (group_by)."
            )
        if plan.distinct and (plan.target or not plan.group_by):
            # "Different X" counts group values. A target question counts that one
            # value's items; with nothing to group by, the different things are the
            # items themselves (repeats dropped by dedup_key).
            plan = plan.model_copy(update={"distinct": False})
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

    # --- the three counters -------------------------------------------------------

    def count_entities(
        self, plan: CountPlan, entities_by_type: dict[str, list[Unit]]
    ) -> CountResult:
        """Distinct entities of the planned types; the graph already holds each once."""
        entities = {unit.id: unit for name in plan.entity_types for unit in entities_by_type[name]}
        matching = list(entities.values())
        if plan.name_contains:
            needle = generate_node_name(plan.name_contains)
            matching = [unit for unit in matching if needle in unit.name]
        return CountResult(
            plan=plan,
            method="graph",
            total=len(matching),
            units=len(entities),
            evidence=[unit.text for unit in matching],
        )

    def count_words(self, plan: CountPlan, units: list[Unit]) -> CountResult:
        """Whole-word, case-sensitive occurrences of the planned spellings in every unit."""
        terms = sorted(set(plan.literal_terms), key=len, reverse=True)
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, terms)) + r")(?!\w)")
        total = 0
        evidence: list[str] = []
        for unit in units:
            for match in pattern.finditer(unit.text):
                total += 1
                window = unit.text[max(match.start() - 60, 0) : match.end() + 60]
                evidence.append(" ".join(window.split()))
        return CountResult(
            plan=plan, method="words", total=total, units=len(units), evidence=evidence
        )

    async def count_by_reading(self, plan: CountPlan, units: list[Unit]) -> CountResult:
        """List matching items from every shard in parallel, then dedupe, merge and tally."""
        shards, tokens_read = self.pack_shards(units)
        semaphore = asyncio.Semaphore(self.max_parallel_calls)

        async def read(shard: list[Unit]) -> tuple[str, ShardItems]:
            async with semaphore:
                return shard[0].id, await self.extract(plan, shard)

        read_shards = await asyncio.gather(*map(read, shards))

        # The model may list one occurrence twice; identical quotes from different
        # units (table rows repeating a value) are different items.
        seen: set[tuple[str, int, str]] = set()
        items: list[ExtractedItem] = []
        for shard_id, shard_items in read_shards:
            for item in shard_items.items:
                marker = (shard_id, item.unit, item.evidence.strip().lower())
                if marker not in seen:
                    seen.add(marker)
                    items.append(item)
        aliases = [names for _, shard_items in read_shards for names in shard_items.aliases]

        names_merged = await self.merge_name_variants(plan, items, aliases)

        if plan.dedup_key:
            keyed: dict[str, ExtractedItem] = {}
            unkeyed: list[ExtractedItem] = []
            for item in items:
                key = _normalize_key(item.key) if item.key else ""
                if not key:
                    unkeyed.append(item)
                elif key not in keyed:
                    keyed[key] = item
            items = [*keyed.values(), *unkeyed]

        # Each item counts 1, or its stated amount when the plan sums a measure.
        def weight(item: ExtractedItem) -> float:
            return (item.amount or 0) if plan.measure else 1

        group_totals: Counter = Counter()
        for item in items:
            if item.group:
                group_totals[item.group] += weight(item)
        groups = group_totals.most_common()
        items_listed = len(items)
        target_names: list[str] = []
        if plan.target:
            # The question's name is matched against the names actually read, so a
            # nickname or partial name finds its person and an unknown name counts zero.
            target_names = await self.match_target(
                plan.target, [name for name, _ in groups], aliases
            )
            items = [item for item in items if item.group in target_names]
        if plan.distinct and not plan.target:
            total: float = len(groups)
        else:
            total = sum(weight(item) for item in items)
        return CountResult(
            plan=plan,
            method="reading",
            total=total,
            units=len(units),
            groups=groups,
            items_listed=items_listed,
            evidence=[
                f"{item.key}: {item.evidence}" if item.key else item.evidence for item in items
            ],
            amounts_missing=(
                sum(1 for item in items if item.amount is None) if plan.measure else 0
            ),
            names_merged=names_merged,
            target_names=target_names,
            llm_calls=len(shards),
            tokens_read=tokens_read,
        )

    # --- reading helpers ------------------------------------------------------------

    def pack_shards(self, units: list[Unit]) -> tuple[list[list[Unit]], int]:
        """Group units into shards of at most ``shard_tokens``; returns (shards, tokens)."""
        shards: list[list[Unit]] = []
        total_tokens = 0
        current: list[Unit] = []
        current_tokens = 0
        for unit in units:
            tokens = len(self.tokenizer.extract_tokens(unit.text))
            if current and current_tokens + tokens > self.shard_tokens:
                shards.append(current)
                current, current_tokens = [], 0
            current.append(unit)
            current_tokens += tokens
            total_tokens += tokens
        if current:
            shards.append(current)
        return shards, total_tokens

    async def extract(self, plan: CountPlan, shard: list[Unit]) -> ShardItems:
        spec = (
            f"Item: {plan.item}\n"
            f"Condition: {plan.condition or 'none'}\n"
            f"Identity attribute (key): {plan.dedup_key or 'none'}\n"
            f"Grouping attribute (group): {plan.group_by or 'none'}\n"
            f"Amount to report (amount): {plan.measure or 'none'}"
        )
        # Unit markers let items be traced back; units are read in full. A
        # document's first line is shown once per shard, before its first unit.
        blocks: list[str] = []
        shown: set[str] = set()
        for index, unit in enumerate(shard):
            block = f"[unit {index}]\n{unit.text}"
            if unit.preamble and unit.preamble not in shown:
                shown.add(unit.preamble)
                block = f"[document start — context only]\n{unit.preamble}\n\n{block}"
            blocks.append(block)
        return await LLMGateway.acreate_structured_output(
            text_input=f"{spec}\n\nSECTION:\n" + "\n\n".join(blocks),
            system_prompt=_read_prompt("broad_extract.txt"),
            response_model=ShardItems,
        )

    async def merge_name_variants(
        self, plan: CountPlan, items: list[ExtractedItem], aliases: list[list[str]]
    ) -> bool:
        """Rewrite every group value to one canonical spelling, in place.

        Returns False when there were too many distinct names to merge.
        """
        # Only group values are names. A dedup key is an identifier (a number, a
        # code, a title): merging "similar" identifiers would join different items.
        names = {item.group for item in items if item.group}
        if len(names) > BROAD_MAX_ALIAS_NAMES:
            return False
        if len(names) < 2:
            return True

        stated = _stated_aliases(aliases)
        text_input = "\n".join(sorted(names))
        if stated:
            text_input += "\n\nStated in the text to be the same:\n" + "\n".join(stated)
        result = await LLMGateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=_read_prompt("broad_merge_names.txt"),
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
        for item in items:
            if item.group:
                item.group = canonical[item.group]
        return True

    async def match_target(
        self, target: str, names: list[str], aliases: list[list[str]]
    ) -> list[str]:
        """The names read from the corpus that are the person or thing ``target`` names.

        Separate from merging: merging joins spellings only when sure, so different
        people stay apart, while the question's name may be a nickname or a partial
        name that only resolves against the names actually present.
        """
        if not names:
            return []
        text_input = f"Name in the question: {target}\n\nNames in the corpus:\n" + "\n".join(names)
        stated = _stated_aliases(aliases)
        if stated:
            text_input += "\n\nStated in the text to be the same:\n" + "\n".join(stated)
        result = await LLMGateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=_read_prompt("broad_match_target.txt"),
            response_model=TargetMatch,
        )
        matched = [name for name in names if name in set(result.names)]
        logger.info("BROAD target %r matched %s", target, matched)
        return matched

    # --- context / completion -------------------------------------------------------

    async def get_context_from_objects(self, query: str, retrieved_objects: CountResult) -> str:
        result = retrieved_objects
        plan = result.plan
        if result.method == "graph":
            how = (
                "Counted by code from the knowledge graph: the distinct entities of type "
                f"{', '.join(plan.entity_types)} ({result.units} in total). Exact."
            )
        elif result.method == "words":
            terms = ", ".join(f'"{term}"' for term in plan.literal_terms)
            how = (
                f"Counted by code: whole-word matches of {terms} in all {result.units} "
                "document chunks / table rows. Exact for these spellings; references by "
                "pronoun or description are not included."
            )
        else:
            how = (
                "Counted by code from the items an LLM listed while reading all "
                f"{result.units} document chunks / table rows ({result.llm_calls} parallel "
                f"calls over {result.tokens_read} tokens). The tally is exact for what was "
                "listed, but reading can miss an item: present it as the count found, not "
                "as a guaranteed exact figure."
            )

        lines = [
            how,
            (
                "Report these numbers; do not recount. If the question does not ask for a "
                "number, answer it from the listed items instead."
            ),
            f"Counted item: {plan.item}",
        ]
        if plan.condition:
            lines.append(f"Condition: {plan.condition}")
        if plan.dedup_key:
            lines.append(f"Repeated mentions of one item removed by: {plan.dedup_key}")
        lines.append(f"TOTAL: {_number(result.total)}")
        if plan.measure:
            lines.append(f"  = the sum of {plan.measure} over the listed items")
            if result.amounts_missing:
                lines.append(f"  ({result.amounts_missing} listed items stated no amount)")
        if plan.target and result.target_names:
            lines.append(
                f"  = the listed items whose {plan.group_by} is "
                f'{" or ".join(result.target_names)} (the names matching "{plan.target}")'
            )
        elif plan.target:
            lines.append(f"  = no listed item has {plan.group_by} {plan.target}")
        elif plan.distinct:
            lines.append(
                f"  = the number of different {plan.group_by} values (spellings of one name "
                f"merged), across {result.items_listed} listed items"
            )
        if result.groups:
            lines.append(f"Tally by {plan.group_by} ({len(result.groups)} groups):")
            lines += [
                f"  {name}: {_number(count)}"
                for name, count in result.groups[:BROAD_MAX_GROUPS_SHOWN]
            ]
            if len(result.groups) > BROAD_MAX_GROUPS_SHOWN:
                lines.append(f"  ... {len(result.groups) - BROAD_MAX_GROUPS_SHOWN} more groups")
        if not result.names_merged:
            lines.append("Note: too many distinct names to merge spelling variants.")
        if plan.list_items and self.listing(result):
            lines.append(
                f"The complete list of the {len(self.listing(result))} counted entries is "
                "appended below your answer by code: do not list them yourself; give the "
                "number and say that the full list follows."
            )
        if result.evidence and plan.target:
            lines.append(f"Listed items of {plan.target} (identifier: quote):")
            lines += [f"  - {entry}" for entry in result.evidence[:BROAD_TARGET_ITEMS_SHOWN]]
        elif result.evidence:
            lines.append("Listed items (quotes):")
            lines += [f'  - "{quote}"' for quote in result.evidence[:BROAD_EVIDENCE_SHOWN]]
        return "\n".join(lines)

    def listing(self, result: CountResult) -> list[str]:
        """What a "list them" question lists: the groups for "different X", else every item."""
        if result.plan.distinct and not result.plan.target:
            return [f"{name} ({_number(count)})" for name, count in result.groups]
        return result.evidence

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any | None = None, **kwargs
    ) -> list[Any]:
        """The LLM phrases the count; a requested list is appended by code, so none is dropped."""
        completions = await super().get_completion_from_context(
            query, retrieved_objects, context=context, **kwargs
        )
        entries = self.listing(retrieved_objects)
        if not retrieved_objects.plan.list_items or not entries:
            return completions
        block = [f"Full list ({len(entries)}):"]
        block += [f"- {entry}" for entry in entries[:BROAD_MAX_LISTED]]
        if len(entries) > BROAD_MAX_LISTED:
            block.append(f"... and {len(entries) - BROAD_MAX_LISTED} more")
        listing = "\n".join(block)
        return [
            f"{completion.rstrip()}\n\n{listing}" if isinstance(completion, str) else completion
            for completion in completions
        ]

    def extract_context_object_ids(self, retrieved_objects: Any) -> dict[str, list[str]] | None:
        return None

    def get_context_evidence(self, retrieved_objects: Any, dataset_id: Any = None):
        return None

    async def append_references(self, completions: list[Any], retrieved_objects: Any) -> list[Any]:
        return completions
