"""BROAD search type: answers "how many" questions by counting in code (SDK-324).

A model cannot count a large corpus by reading it: it skims, and a count needs
every record, not the most similar ones. BROAD never asks a model for a number.
One LLM call plans the count, then the cheapest counter that can answer runs:

- graph:   distinct things of a type the graph holds ("how many people") are its
           entity nodes, optionally filtered by name. Exact, no LLM calls.
- words:   how many times a word is written is a whole-word match over every
           chunk. Exact for the planned spellings, no LLM calls.
- records: documents that are a table (CSV, JSON records, an email archive) or
           record lines (a log, a templated report) are parsed by code; one LLM
           call maps the question onto the columns or writes line expressions,
           and code evaluates them over every record (broad_table.py). Exact.
- reading: everything else. Every chunk is read in parallel calls; each LISTS the
           matching items with a quote, and code merges name variants, drops
           repeated mentions and tallies. The tally is exact over what was listed;
           an item the reading missed is not in it, and the answer says so.

The final LLM call only phrases the computed numbers."""

import asyncio
import re
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.engine.utils import generate_node_name
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.broad_table import (
    LINE_COLUMNS,
    LineMatcher,
    LineQuery,
    LineQueryError,
    QueryRun,
    ShapedLines,
    ShapeStats,
    Table,
    TableQuery,
    TableSummary,
    describe,
    describe_shapes,
    parse_table,
    record_lines,
    render_row,
)
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.shared.logging_utils import get_logger

logger = get_logger("BroadRetriever")


# Small units (table rows, short chunks) are packed into one reading call up to
# this many tokens. A chunk is never split: one larger than this is read whole,
# on its own, so the smallest call is one full chunk as ingestion stored it.
BROAD_SHARD_TOKENS = 2_000
BROAD_MAX_PARALLEL_CALLS = 16
# Megabytes of text held in memory at once: the dataset is fetched in batches of whole
# documents of about this size. Scale it to the machine (retriever_specific_config).
BROAD_BATCH_MB = 256
# The most tokens the reading counter may send to the LLM for one question: about 5,000
# pages. A question that needs more read raises instead (retriever_specific_config).
BROAD_MAX_READING_TOKENS = 20_000_000
# A reading call that hangs holds the whole wave: one call stalled for 84 and then
# for 168 minutes. A call is retried once after this long, then the search fails.
BROAD_CALL_TIMEOUT_SECONDS = 300
# Graph node types whose text is read: document chunks and table rows.
BROAD_TEXT_NODE_TYPES = ("DocumentChunk", "DltRow")
# Longest document first line (a CSV header, a title) repeated as shard context.
BROAD_PREAMBLE_CHARS = 400
# Longest tail of the previous chunk shown as context before a chunk read without
# it: the record a chunk boundary cut in two keeps the heading that names it.
BROAD_CONTEXT_CHARS = 1_500
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
# Passages where the question's name itself appears, shown to the target match: where
# records name a person by an id, code or handle, the name is declared elsewhere (a
# roster, a users file, a legend), and the match needs that declaration.
BROAD_TARGET_PASSAGES = 6
BROAD_TARGET_PASSAGE_CHARS = 160
_ENTITY_DESCRIPTION_CHARS = 200
# The planner sees this much of the corpus (from its start, middle and end) so it names an
# identity the text actually writes: "Recommendation 3", "H.R. 4418", "INV-2024-00137".
BROAD_SAMPLE_EXCERPTS = 3
BROAD_SAMPLE_CHARS = 700


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
    # A relation between two things of the same kind (a connection, a co-authorship),
    # listed once per participant: its identity is (participant, other), not the key alone.
    relation: bool = False
    # The question counts items in effect, and a later event can end one (a connection
    # removed, an order cancelled after it was placed): such events subtract the item.
    reversible: bool = False
    # With reversible: the question asks how many items are now in the ended state
    # ("how many tickets are resolved") rather than still in effect ("still open").
    counts_ended: bool = False
    # "What share of tickets were escalated": the item is every ticket; this condition
    # marks the subset, and the answer is subset over all, both counted by code.
    ratio_condition: str | None = None
    # "Average order value": the measure's mean over the items instead of its sum.
    average: bool = False
    # Why the question cannot be answered by listing what sections of text show
    # (absence across the corpus, a comparison of separate totals); nothing is read.
    unsupported: str | None = None


class ExtractedItem(BaseModel):
    unit: int
    key: str | None = None
    group: str | None = None
    # Every value of the grouping attribute when there are several (all authors of a
    # paper; both members of a connection): code makes one entry per value.
    groups: list[str] = []
    # The item's value of the plan's measure, when the plan sums one.
    amount: float | None = None
    # The text says this item was later undone (removed, cancelled, returned, revoked):
    # code drops the item with the same group and key instead of counting this entry.
    undone: bool = False
    # The entry's date as written, YYYY-MM-DD, when the section gives one: for an item
    # whose state changes over time, the latest entry decides.
    when: str | None = None
    # Whether the item meets the plan's ratio condition, when the plan has one.
    matches: bool | None = None
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
    # The previous chunk of the same document (its text, shared, not copied), whose end
    # is shown as context when that chunk is not read in the same call; and its id.
    previous_text: str = ""
    previous_id: str = ""
    # The document the chunk belongs to; empty for a table row.
    document: str = ""


@dataclass
class CountResult:
    plan: CountPlan
    method: Literal["graph", "words", "reading", "table", "unsupported"]
    total: float
    units: int
    # For a plan with a ratio condition: the number of all items, the total being the subset.
    denominator: int = 0
    groups: list[tuple[str, float]] = field(default_factory=list)
    # Every counted item as one line; the answer context shows a sample, a listing shows all.
    evidence: list[str] = field(default_factory=list)
    # For a plan with a measure: listed items that stated no amount (added as 0).
    amounts_missing: int = 0
    # For a plan with a target: the corpus names matched to it; empty when none matched.
    target_names: list[str] = field(default_factory=list)
    # For a plan with a dedup key: counted entries that carried no key, so could not be
    # checked for repeats. The most the total could be over by.
    unkeyed: int = 0
    names_merged: bool = True
    items_listed: int = 0
    # Entries the reading calls returned before repeated mentions were removed: the
    # gap to items_listed is what dedup took out, and the answer states it.
    entries_read: int = 0
    llm_calls: int = 0
    tokens_read: int = 0
    # For the table method: the query code evaluated over the parsed rows.
    table_query: TableQuery | None = None
    # What part of the corpus the count covered, when not all of it.
    scope: str = ""


def _read_prompt(name: str) -> str:
    prompt = read_query_prompt(name)
    if prompt is None:
        raise FileNotFoundError(f"BROAD prompt {name!r} could not be read.")
    return prompt


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:,.2f}"


# What the model writes between names when it returns a stated alias as one string
# ("Pavel Horák is Pav", "Robert, known as Bob", "Maria Duarte (Maria)").
_ALIAS_JOINERS = re.compile(
    r"\s*(?:[,;()=]|\bis\b|\b(?:usually |often |also )?(?:known as|called|aka)\b)\s*", re.IGNORECASE
)


def _alias_groups(raw: list[list[str]]) -> list[list[str]]:
    """Alias groups as lists of names; an entry written as a sentence is split into its names."""
    groups = []
    for group in raw:
        names = [name.strip() for entry in group for name in _ALIAS_JOINERS.split(entry)]
        names = [name for name in names if name]
        if len(names) > 1:
            groups.append(names)
    return groups


def _stated_aliases(aliases: list[list[str]]) -> list[str]:
    return sorted({" = ".join(group) for group in aliases})


def _match_stated(aliases: list[list[str]], names: set[str]) -> list[list[str]]:
    """Each stated alias group as the read names it covers, matched by loose spelling."""
    by_spelling: dict[str, list[str]] = {}
    for name in names:
        by_spelling.setdefault(_loose_name(name), []).append(name)
    return [
        [name for alias in group for name in by_spelling.get(_loose_name(alias), [])]
        for group in aliases
    ]


def _normalize_key(value: str) -> str:
    """One item, one key: "#20142", "20142" and "no. 20142" are the same identifier,
    and so are "PR-42", "PR #42" and "pr 42"; but "PR 42" and "issue 42" are two
    items when a question spans several kinds of contribution.

    A key made of one word label and digits is that label plus its digit runs; a bare
    identifier is its digits; a key made of several words ("Matchday 5, Harbour City
    v Glenmarsh City") keeps every word, because dropping the words would fold every
    record that shares the number into one.
    """
    tokens = re.findall(r"[0-9a-z]+", value.lower())
    if tokens and tokens[0] in ("no", "number", "nr"):
        tokens = tokens[1:]
    words = [token for token in tokens if token.isalpha()]
    if not words:
        return "-".join(tokens)
    if len(words) == 1 and any(token.isdigit() for token in tokens):
        return "-".join(words + [token for token in tokens if token.isdigit()])
    return "-".join(tokens)


def _spelling_pattern(term: str) -> str:
    """A regex for one planned spelling: any whitespace between its words, and any case
    when it is written in lower case."""
    body = r"\s+".join(map(re.escape, term.split()))
    return body if term != term.lower() else f"(?i:{body})"


def _corpus_sample(units: list[Unit]) -> str:
    """Short excerpts from the start, middle and end of the corpus, in document order."""
    if not units:
        return ""
    picks = sorted(
        {
            round(i * (len(units) - 1) / max(BROAD_SAMPLE_EXCERPTS - 1, 1))
            for i in range(BROAD_SAMPLE_EXCERPTS)
        }
    )
    return "\n---\n".join(units[i].text[:BROAD_SAMPLE_CHARS].strip() for i in picks)


def _document_texts(units: list[Unit]) -> Iterator[str] | None:
    """Each document's text, rebuilt from its chunks one document at a time; None when a
    unit has no document."""
    if not units or any(not unit.document for unit in units):
        return None
    # Units of a document are contiguous and in chunk order. Chunks partition their
    # document exactly and may cut mid-line: joined with nothing, they are the document
    # again, and a record cut in two is whole.
    return (
        "".join(unit.text for unit in parts)
        for _, parts in groupby(units, key=lambda unit: unit.document)
    )


def _groups_are_keys(items: list[ExtractedItem]) -> bool:
    """Whether the group values are the items' own identifiers: most items that carry both
    a group and a key write the key inside the group ("H.R. 4418" / "4418")."""
    both = [item for item in items if item.key and item.group]
    carried = sum(
        1 for item in both if _normalize_key(item.key or "") in _normalize_key(item.group or "")
    )
    return bool(both) and carried * 2 > len(both)


def _same_noun(item: str, attribute: str) -> bool:
    """Whether a grouping attribute names the item itself ("an incident" / "incident")."""

    def head(phrase: str) -> str:
        words = re.findall(r"[a-z]+", phrase.lower())
        words = [w for w in words if w not in ("a", "an", "the", "each", "every")]
        return words[-1].rstrip("s") if words else ""

    return bool(head(attribute)) and head(item) == head(attribute)


def _tail(text: str) -> str:
    """The final paragraph of a chunk (its last record, whole or cut), capped."""
    paragraph = re.split(r"\n\s*\n", text.rstrip())[-1]
    return paragraph[-BROAD_CONTEXT_CHARS:]


def _loose_name(name: str) -> str:
    """A name keeps its letters and digits ("raj921" is not "921"); only @ and punctuation go."""
    return re.sub(r"[^0-9a-z]+", "", name.lstrip("@").lower())


def _passages_naming(target: str, units: list["Unit"]) -> list[str]:
    """Short passages, from distinct units, in which the target's name is written out."""
    words = [re.escape(word) for word in target.lstrip("@").split()]
    if not words:
        return []
    pattern = re.compile(r"(?<!\w)" + r"\W+".join(words) + r"(?!\w)", re.IGNORECASE)
    passages: list[str] = []
    for unit in units:
        match = pattern.search(unit.text)
        if match:
            start = max(match.start() - BROAD_TARGET_PASSAGE_CHARS, 0)
            end = match.end() + BROAD_TARGET_PASSAGE_CHARS
            passages.append(" ".join(unit.text[start:end].split()))
            if len(passages) >= BROAD_TARGET_PASSAGES:
                break
    return passages


def _drop_attribute_named_groups(
    items: list[ExtractedItem], group_by: str | None
) -> list[ExtractedItem]:
    """A group written as the attribute's own word ("guest" under group_by "guest") is no
    value: the entry keeps counting, unattributed, instead of building a phantom group."""
    if not group_by:
        return items
    names = {group_by.lower(), group_by.lower().rstrip("s")}
    for item in items:
        if item.group and item.group.strip().lower().rstrip("s") in names:
            item.group = None
    return items


def _one_entry_per_group(items: list[ExtractedItem], relation: bool) -> list[ExtractedItem]:
    """Expand an item listed once with all its group values into one entry per value.

    A paper with four authors is one item of each author. A relation (a connection,
    a co-authorship) belongs to every participant: for each participant one entry
    keyed by each other participant, so "Helga accepted a request from Arthur"
    counts for Arthur too. The model lists the item once; code does the pairing.
    """
    expanded: list[ExtractedItem] = []
    for item in items:
        values = [v for v in item.groups if v] or ([item.group] if item.group else [])
        if relation:
            if item.key and item.key not in values:
                values.append(item.key)
            if len(values) < 2:
                continue  # a relation needs two participants
            for participant in values:
                for other in values:
                    if other != participant:
                        expanded.append(
                            item.model_copy(update={"group": participant, "key": other})
                        )
        elif values:
            expanded += [
                item.model_copy(update={"group": value}) for value in dict.fromkeys(values)
            ]
        else:
            expanded.append(item)
    return expanded


def _adopt_labels(keyed: list[tuple[str, ExtractedItem]]) -> list[tuple[str, ExtractedItem]]:
    """A bare number takes the kind of the labelled key with the same digits.

    "PR 42" and "issue 42" are two items, but "1032" written once bare and once as
    "PR 1032" is one: a bare number never contradicts a kind. Only when exactly one
    kind carries those digits in the same group is the adoption unambiguous.
    """
    labelled: dict[tuple[str, str], set[str]] = {}
    for key, item in keyed:
        if not key.isdigit() and re.search(r"\d", key):
            labelled.setdefault((item.group or "", key.rsplit("-", 1)[-1]), set()).add(key)
    adopted = []
    for key, item in keyed:
        kinds = labelled.get((item.group or "", key)) if key.isdigit() else None
        adopted.append((next(iter(kinds)) if kinds and len(kinds) == 1 else key, item))
    return adopted


def _is_label(name: str) -> bool:
    """A handle or an all-capitals form ("@ann", "ORTIZ:") is how a name is rendered in a
    header or a speaker label, not a spelling of it."""
    return name.startswith("@") or (name.isupper() and any(c.isalpha() for c in name))


def _canonical_spelling(names: list[str], used: Counter) -> str:
    """The spelling a group of variants is tallied under: a name over a label form, then
    the one the corpus uses most (so a nickname does not stand in for the name), then the
    longest. A transcript writes "ORTIZ:" on every turn and "Dr. Lena Ortiz" once; the
    tally reads the name."""
    return min(names, key=lambda name: (_is_label(name), -used[name], -len(name), name))


def _line_query_problem(query: LineQuery) -> str | None:
    """What makes a line query unusable before it runs: an aggregate over values that it
    never captures."""
    if query.answerable and query.aggregate != "count_rows" and not query.value_regex:
        return f"aggregate {query.aggregate} needs a value_regex that captures the value"
    if query.answerable and (query.group or query.target) and not query.value_regex:
        return "grouping or a target needs a value_regex that captures the value"
    return None


def _table_context(result: CountResult) -> str:
    """What the answer model is told about a count code made over parsed rows."""
    query = result.table_query
    assert query is not None
    lines = [
        (
            f"Counted by code over all {result.units} rows of the table the documents hold. "
            "Exact. Report these numbers; do not recount."
        ),
    ]
    for rule in query.filters:
        value = f" {rule.value!r}" if rule.value else ""
        lines.append(f"Filter: {rule.column} {rule.op}{value}")
    if query.target:
        found = " or ".join(result.target_names) or "no row"
        lines.append(f"Rows whose {query.group_by} is {query.target!r}: matched {found}")
    measure = query.aggregate.replace("_", " ") + (f" of {query.column}" if query.column else "")
    lines.append(f"TOTAL ({measure}): {_number(result.total)}")
    if result.groups:
        lines.append(f"Tally by {query.group_by} ({len(result.groups)} groups):")
        lines += [
            f"  {name}: {_number(count)}" for name, count in result.groups[:BROAD_MAX_GROUPS_SHOWN]
        ]
    if result.plan.list_items and result.evidence:
        lines.append(
            f"The complete list of the {result.items_listed} matching rows is appended below "
            "your answer by code: do not list them yourself."
        )
    elif result.evidence:
        lines.append("Matching rows (first ones):")
        lines += [f"  - {row}" for row in result.evidence[:BROAD_EVIDENCE_SHOWN]]
    return "\n".join(lines)


def _provenance_note(result: CountResult) -> str:
    """One line saying how a reading count was obtained, stated by code in every answer: a
    reading count is exact over what was found, and reading can miss or repeat an item."""
    if result.method == "table":
        scope = f"; {result.scope}" if result.scope else ""
        return f"(Counted by code over all {result.units} records in the documents{scope}.)"
    if result.method != "reading":
        return ""
    note = (
        f"(Counted by code from the items found while reading all {result.units} chunks; "
        "an item the reading missed is not included."
    )
    if result.unkeyed:
        note += (
            f" {result.unkeyed} entries whose identifier could not be read were left out, so "
            f"the total may be under by up to {result.unkeyed}."
        )
    return note + ")"


class BroadLimitError(ValueError):
    """The dataset is larger than a BROAD limit allows; the message names the setting."""


def _units_of(nodes: list, edges: list) -> list[Unit]:
    """The chunks and table rows among graph nodes as units, in document order.

    One unit per chunk, as ingestion stored it: a chunk is never split or rejoined. A
    chunk from the middle of a document carries that document's first line (a CSV
    header lives only in chunk 0) and the previous chunk's text (a record cut in two
    keeps the heading that names it).
    """
    document_of = {
        str(source): str(target)
        for source, target, relationship_name, _ in edges
        if relationship_name == "is_part_of"
    }
    parts_of: dict[str, list[tuple[int, str, str]]] = {}
    for node_id, props in nodes:
        document = document_of.get(str(node_id))
        if props.get("type") in BROAD_TEXT_NODE_TYPES and props.get("text") and document:
            index = int(props.get("chunk_index") or 0)
            parts_of.setdefault(document, []).append((index, str(node_id), props["text"]))
    units: list[Unit] = []
    for document, parts in sorted(parts_of.items()):
        parts.sort()
        first_line = parts[0][2].strip().split("\n", 1)[0][:BROAD_PREAMBLE_CHARS]
        previous: tuple[str, str] | None = None
        for _, node_id, text in parts:
            units.append(
                Unit(
                    id=node_id,
                    text=text,
                    preamble="" if first_line in text else first_line,
                    previous_text=previous[1] if previous else "",
                    previous_id=previous[0] if previous else "",
                    document=document,
                )
            )
            previous = (node_id, text)
    return units


@dataclass
class Corpus:
    """A dataset's documents, fetched from the graph in batches of whole documents so the
    text in memory at once stays near ``batch_chars``, whatever the dataset's size.

    Batches are planned from the stored size of each document (its file size), scaled by
    how much text a byte turned out to hold in the batches already fetched. A corpus that
    fits in one batch is fetched once and kept. A single document larger than the budget
    cannot be split (its records can span chunks), and raises.
    """

    graph: Any
    documents: list[str]
    sizes: dict[str, int]
    batch_chars: int
    # Characters of text in the corpus, known once every batch has been fetched.
    chars: int = 0
    kept: list[Unit] | None = None

    @classmethod
    def of(cls, units: list[Unit]) -> "Corpus":
        """A corpus already in memory, as one batch."""
        chars = sum(len(unit.text) for unit in units)
        return cls(graph=None, documents=[], sizes={}, batch_chars=chars, chars=chars, kept=units)

    def fits_one_batch(self) -> bool:
        known = [self.sizes.get(document, 0) for document in self.documents]
        return all(known) and sum(known) <= self.batch_chars

    async def fetch(self, documents: list[str]) -> list[Unit]:
        nodes, edges = await self.graph.get_neighborhood(documents, depth=1)
        return _units_of(nodes, edges)

    async def batches(self) -> AsyncIterator[list[Unit]]:
        if self.kept is not None:
            yield self.kept
            return
        if self.fits_one_batch():
            self.kept = await self.fetch(self.documents)
            self.chars = sum(len(unit.text) for unit in self.kept)
            yield self.kept
            return
        ratio, chars, start = 1.0, 0, 0
        while start < len(self.documents):
            batch = [self.documents[start]]
            planned = self.sizes.get(batch[0], 0) * ratio
            while start + len(batch) < len(self.documents):
                size = self.sizes.get(self.documents[start + len(batch)])
                if not size or planned + size * ratio > self.batch_chars:
                    break
                batch.append(self.documents[start + len(batch)])
                planned += size * ratio
            units = await self.fetch(batch)
            text = sum(len(unit.text) for unit in units)
            if len(batch) == 1 and text > self.batch_chars:
                raise BroadLimitError(
                    f"BROAD: document {batch[0]} holds {text:,} characters of text, more than "
                    f"one batch of {self.batch_chars:,} (batch_mb). Raise batch_mb in "
                    "retriever_specific_config to what this machine can hold."
                )
            stored = sum(self.sizes.get(document, 0) for document in batch)
            if stored:
                ratio = max(ratio, text / stored)
            chars += text
            start += len(batch)
            logger.info(
                "BROAD batch: %d documents, %d chunks, %d characters", len(batch), len(units), text
            )
            yield units
        self.chars = chars

    async def sample(self) -> list[Unit]:
        """Units from the corpus's start, middle and end, for the planner: every unit
        when the corpus fits one batch, else a unit of the first, middle and last document."""
        if self.kept is not None or self.fits_one_batch():
            async for units in self.batches():
                return units
        picks = sorted({0, len(self.documents) // 2, len(self.documents) - 1})
        sample: list[Unit] = []
        for index, pick in enumerate(picks):
            units = await self.fetch([self.documents[pick]])
            if units:
                sample.append(units[[0, len(units) // 2, -1][index] if len(picks) == 3 else 0])
        return sample


class BroadRetriever(CompletionRetriever):
    """Count-and-aggregate search over every unit of a dataset.

    Settings (all optional, passed through ``retriever_specific_config``):

    - ``shard_tokens``: packing budget for small units (table rows, short chunks); a
      stored chunk larger than it is read whole, alone, never split.
    - ``max_parallel_calls``: reading calls in flight at once.
    - ``call_timeout``: seconds one reading call may take before it is retried once.
    - ``batch_mb``: megabytes of text held in memory at once. The dataset is fetched in
      batches of whole documents of about this size, so any dataset size can be counted;
      raise it on a machine with more memory, lower it on a smaller one. A document
      larger than one batch raises.
    - ``max_reading_tokens``: the most text the reading counter may send to the LLM. A
      question that needs every chunk read over a larger dataset raises instead of
      reading (cost and time grow with the dataset); counts made by code have no limit.
    """

    def __init__(
        self,
        shard_tokens: int = BROAD_SHARD_TOKENS,
        call_timeout: float = BROAD_CALL_TIMEOUT_SECONDS,
        max_parallel_calls: int = BROAD_MAX_PARALLEL_CALLS,
        batch_mb: float = BROAD_BATCH_MB,
        max_reading_tokens: int = BROAD_MAX_READING_TOKENS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if batch_mb <= 0 or max_reading_tokens <= 0:
            raise ValueError("BROAD batch_mb and max_reading_tokens must be positive")
        self.shard_tokens = shard_tokens
        self.call_timeout = call_timeout
        self.max_parallel_calls = max_parallel_calls
        self.batch_chars = int(batch_mb * 1_000_000)
        self.max_reading_tokens = max_reading_tokens
        self.tokenizer = TikTokenTokenizer()

    async def get_retrieved_objects(self, query: str) -> CountResult:
        graph_engine = (await get_unified_engine()).graph
        entity_types = await self.load_entity_types(graph_engine)
        corpus = await self.open_corpus(graph_engine)
        plan = await self.plan(query, entity_types, await corpus.sample())
        logger.info("BROAD plan: %s", plan.model_dump())

        result: CountResult | None = None
        if plan.unsupported:
            result = CountResult(plan=plan, method="unsupported", total=0, units=0)
        elif plan.source == "entities":
            entities = await self.load_entities(
                graph_engine, {name: entity_types[name] for name in plan.entity_types}
            )
            result = self.count_entities(plan, entities)
        elif not corpus.documents and corpus.kept is None:
            raise NoDataError("No data found in the system, please add data first.")
        elif (
            plan.literal_terms
            and plan.dedup_key is None
            and not (plan.condition or plan.group_by or plan.measure)
        ):
            # Word matches count occurrences; a keyed item ("reviews that mention film")
            # counts things that hold a match, which the other lanes do.
            result = await self.count_words(plan, corpus)
        else:
            result = await self.count_records(query, plan, corpus)
        if result is None:
            result = await self.count_by_reading(plan, corpus)

        logger.info(
            "BROAD %s count: total=%d over %d units, %d LLM calls, %d tokens read",
            result.method,
            result.total,
            result.units,
            result.llm_calls,
            result.tokens_read,
        )
        return result

    async def count_records(
        self, query: str, plan: CountPlan, corpus: Corpus
    ) -> CountResult | None:
        """Count by code when the documents are records: a table (delimited, JSON, email)
        or record lines (a log, a templated report). None when they are not, or when the
        records cannot answer the question; reading then decides.

        One pass over the corpus finds the tables and summarizes them (or counts the line
        shapes of the rest); after the query call, a second pass evaluates the query.
        Neither pass holds more than one batch of documents."""
        survey = await self._survey(corpus)
        if survey is None:
            return None
        summary, others, stats = survey
        if summary is not None:
            result = await self.count_table(query, plan, summary, corpus)
            if result is not None and others:
                result.scope = (
                    f"only the table in the documents; {others} other documents "
                    "(not tables) were not counted"
                )
            return result
        shaped = stats.result()
        return await self.count_lines(query, plan, shaped, corpus) if shaped else None

    async def _survey(self, corpus: Corpus) -> tuple[TableSummary | None, int, ShapeStats] | None:
        """One pass over the corpus: the summary of its tables and how many documents are
        not tables, or, when no two tables agree on columns, the shapes of every line.
        Each document is parsed, summarized and let go (parsed rows take several times the
        memory of their text; the query pass parses them again). None when a unit has
        no document."""
        summary: TableSummary | None = None
        mixed = False
        others = 0
        stats = ShapeStats()
        async for units in corpus.batches():
            texts = _document_texts(units)
            if texts is None:
                return None
            for text in texts:
                table = parse_table(text)
                if table is None:
                    others += 1
                    stats.add(record_lines(text))
                    continue
                if summary is None:
                    summary = TableSummary(columns=table.columns)
                mixed = mixed or table.columns != summary.columns
                summary.add(table.rows)
        if summary is not None and not mixed:
            return summary, others, stats
        if mixed:
            # Tables that disagree on their columns are counted as lines, all of them.
            stats = ShapeStats()
            async for units in corpus.batches():
                for text in _document_texts(units) or []:
                    stats.add(record_lines(text))
        return None, others, stats

    async def _ask(self, text_input: str, prompt: str, response_model: type[Any]) -> Any:
        """One structured LLM call. A call that hangs holds the whole search (one stalled
        for 84 and then 168 minutes): past ``call_timeout`` it is retried once, then the
        search fails."""
        for attempt in (1, 2):
            try:
                return await asyncio.wait_for(
                    LLMGateway.acreate_structured_output(
                        text_input=text_input,
                        system_prompt=_read_prompt(prompt),
                        response_model=response_model,
                    ),
                    self.call_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "BROAD %s call timed out after %.0fs (attempt %d)",
                    prompt,
                    self.call_timeout,
                    attempt,
                )
        raise TimeoutError(f"BROAD: a {prompt} call timed out twice")

    # --- sources --------------------------------------------------------------

    async def load_entity_types(self, graph_engine) -> dict[str, str]:
        """Every entity type in the graph: its name and node id. Its entities are loaded
        only when the plan counts them."""
        nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["EntityType"]}])
        return {props["name"]: str(node_id) for node_id, props in nodes if props.get("name")}

    async def load_entities(self, graph_engine, types: dict[str, str]) -> dict[str, list[Unit]]:
        """The entities of the given types (name to type node id), grouped by type name."""
        nodes, edges = await graph_engine.get_neighborhood(list(types.values()), depth=1)
        props_by_id = {str(node_id): props for node_id, props in nodes}
        name_of = {type_id: name for name, type_id in types.items()}
        entities_by_type: dict[str, list[Unit]] = {name: [] for name in types}
        for source_id, target_id, relationship_name, _ in edges:
            entity = props_by_id.get(str(source_id), {})
            if relationship_name != "is_a" or entity.get("type") != "Entity":
                continue
            if str(target_id) not in name_of:
                continue
            description = (entity.get("description") or "")[:_ENTITY_DESCRIPTION_CHARS]
            entities_by_type[name_of[str(target_id)]].append(
                Unit(
                    id=str(source_id),
                    text=f"{entity['name']}: {description}",
                    name=entity["name"],
                )
            )
        return entities_by_type

    async def open_corpus(self, graph_engine) -> Corpus:
        """The dataset's documents (ids and stored sizes); their text is fetched in batches.

        Listed from the graph, not found by vector search: a count needs every unit, and
        similarity plays no part in which units exist."""
        document_types = [cls.__name__ for cls in get_all_subclasses(Document)]
        nodes, _ = await graph_engine.get_filtered_graph_data([{"type": document_types}])
        documents = sorted(str(node_id) for node_id, _ in nodes)
        sizes = await self.document_sizes(documents)
        logger.info(
            "BROAD corpus: %d documents, %d bytes stored", len(documents), sum(sizes.values())
        )
        return Corpus(
            graph=graph_engine, documents=documents, sizes=sizes, batch_chars=self.batch_chars
        )

    async def document_sizes(self, documents: list[str]) -> dict[str, int]:
        """Each document's stored size in bytes (a document's graph id is its data id)."""
        engine = get_relational_engine()
        sizes: dict[str, int] = {}
        async with engine.get_async_session() as session:
            for start in range(0, len(documents), 5_000):
                ids = [UUID(document) for document in documents[start : start + 5_000]]
                rows = await session.execute(
                    select(Data.id, Data.data_size).where(Data.id.in_(ids))
                )
                sizes.update({str(data_id): size or 0 for data_id, size in rows})
        return sizes

    # --- planning ---------------------------------------------------------------

    async def plan(
        self, query: str, entity_types: dict[str, str], units: list[Unit] | None = None
    ) -> CountPlan:
        type_list = "\n".join(
            f"- {name}" for name in sorted(entity_types)[:BROAD_MAX_PLANNER_TYPES]
        )
        text_input = f"Question: {query}\n\nEntity types in the graph:\n{type_list or '(none)'}"
        sample = _corpus_sample(units or [])
        if sample:
            text_input += (
                f"\n\nExcerpts of the corpus (for how it writes things; not all of it):\n{sample}"
            )
        plan = await self._ask(
            text_input,
            "broad_plan.txt",
            CountPlan,
        )
        if plan.target and not plan.group_by:
            # A target is a value of some attribute; one retry names the omission.
            plan = await self._ask(
                (
                    f"{text_input}\n\nYour previous plan named the target {plan.target!r} "
                    "without group_by. Set group_by to the attribute that value belongs to "
                    "(a reagent, a cell line, a country, an assignee) and return the full plan."
                ),
                "broad_plan.txt",
                CountPlan,
            )
        if plan.dedup_key and re.search(r"\bor\b", plan.dedup_key):
            # "The title or number" matches nothing when one mention gives the title and
            # another the number: one retry asks for a single identifier.
            plan = await self._ask(
                (
                    f"{text_input}\n\nYour previous dedup_key {plan.dedup_key!r} names "
                    "alternatives. Name ONE identifier that the excerpts show every item has "
                    "(its number or code when it has one), and return the full plan."
                ),
                "broad_plan.txt",
                CountPlan,
            )
        if plan.target and not plan.group_by:
            raise ValueError(
                f"BROAD planner named a target ({plan.target!r}) without the attribute it is a "
                "value of (group_by)."
            )
        if plan.distinct and plan.group_by and _same_noun(plan.item, plan.group_by):
            # "How many distinct incidents": grouping incidents by the incident is the
            # item count itself, and dedup_key already counts each once.
            plan = plan.model_copy(update={"group_by": None, "distinct": False})
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
            unknown = [name for name in plan.entity_types if name not in entity_types]
            if unknown or not plan.entity_types:
                raise ValueError(f"BROAD planner chose entity types not in the graph: {unknown}")
        return plan

    # --- the counters ------------------------------------------------------------

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

    async def count_words(self, plan: CountPlan, corpus: Corpus) -> CountResult:
        """Whole-word occurrences of the planned spellings in every unit.

        A spelling in lower case matches any case ("gross margin" also finds the
        sentence-initial "Gross margin"); one with capitals matches as written, so a
        name stays a name. The words of a phrase may be split by a line break.
        """
        terms = sorted(set(plan.literal_terms), key=len, reverse=True)
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(_spelling_pattern, terms)) + r")(?!\w)")
        total = units_read = 0
        evidence: list[str] = []
        async for units in corpus.batches():
            units_read += len(units)
            for unit in units:
                for match in pattern.finditer(unit.text):
                    total += 1
                    if len(evidence) < BROAD_MAX_LISTED:
                        window = unit.text[max(match.start() - 60, 0) : match.end() + 60]
                        evidence.append(" ".join(window.split()))
        return CountResult(
            plan=plan, method="words", total=total, units=units_read, evidence=evidence
        )

    async def count_lines(
        self, query: str, plan: CountPlan, shaped: ShapedLines, corpus: Corpus
    ) -> CountResult | None:
        """Answer over record-like lines (a log, a templated report): one LLM call sees every
        line shape and writes expressions that select the lines and capture the value; code
        runs them over every line, batch by batch. None when the lines cannot answer."""
        text_input = f"Question: {query}\n\nLine shapes:\n{describe_shapes(shaped)}"
        line_query = await self._ask(
            text_input,
            "broad_line_query.txt",
            LineQuery,
        )
        problem = _line_query_problem(line_query)
        run = None
        if problem is None and line_query.answerable:
            try:
                run = await self._run_lines(line_query, corpus)
                if run.table_rows == 0:
                    problem = f"line_regex matched none of the {shaped.lines} lines"
            except LineQueryError as error:
                problem = str(error)
        if problem:
            # One retry, shown what went wrong and five real lines spread over the corpus.
            step = max(shaped.lines // 5, 1)
            samples = await self._sample_lines(corpus, step)
            line_query = await self._ask(
                f"{text_input}\n\nYour previous expressions failed: {problem}. "
                "Real lines, exactly as written:\n"
                + "\n".join(samples[:5])
                + "\n\nReturn corrected expressions.",
                "broad_line_query.txt",
                LineQuery,
            )
            run = None
            if line_query.answerable and _line_query_problem(line_query) is None:
                try:
                    run = await self._run_lines(line_query, corpus)
                except LineQueryError:
                    run = None
        logger.info("BROAD line query: %s", line_query.model_dump())
        if run is None or run.table_rows == 0:
            return None  # nothing selected: reading decides
        answer = run.answer()
        return CountResult(
            plan=plan.model_copy(update={"list_items": plan.list_items or line_query.list_rows}),
            method="table",
            total=answer.total,
            units=shaped.lines,
            groups=answer.groups,
            evidence=[row[1] for row in answer.matched],
            target_names=answer.target_values,
            items_listed=answer.matched_count,
            llm_calls=1,
            table_query=run.query,
        )

    async def _run_lines(self, line_query: LineQuery, corpus: Corpus) -> QueryRun:
        """Run a line query over every record line of the corpus: the lines its
        expressions select are the rows (captured values, line) of a table query."""
        matcher = LineMatcher(line_query)
        valued = line_query.value_regex is not None
        table_query = TableQuery(
            answerable=True,
            aggregate=line_query.aggregate,
            column="value" if valued and line_query.aggregate != "count_rows" else None,
            group_by="value" if valued and (line_query.group or line_query.target) else None,
            target=line_query.target,
            list_rows=line_query.list_rows,
        )
        run = QueryRun(LINE_COLUMNS, table_query, keep_rows=BROAD_MAX_LISTED)
        async for units in corpus.batches():
            for text in _document_texts(units) or []:
                run.add(matcher.rows(record_lines(text)))
        return run

    async def _sample_lines(self, corpus: Corpus, step: int) -> list[str]:
        """Every ``step``-th record line of the corpus."""
        samples: list[str] = []
        seen = 0
        async for units in corpus.batches():
            for text in _document_texts(units) or []:
                lines = record_lines(text)
                samples += [line for i, line in enumerate(lines, seen) if i % step == 0]
                seen += len(lines)
        return samples

    async def count_table(
        self,
        query: str,
        plan: CountPlan,
        summary: TableSummary,
        corpus: Corpus,
    ) -> CountResult | None:
        """Answer over parsed rows: one LLM call maps the question onto the columns, code
        evaluates it over every row of the corpus's tables, parsed batch by batch. None
        when the columns cannot answer the question."""
        table_query = await self._ask(
            f"Question: {query}\n\nTable:\n{describe(summary)}",
            "broad_table_query.txt",
            TableQuery,
        )
        logger.info("BROAD table query: %s", table_query.model_dump())
        if not table_query.answerable:
            return None
        try:
            run = QueryRun(summary.columns, table_query, keep_rows=BROAD_MAX_LISTED)
        except KeyError as error:
            logger.warning("BROAD table query unusable (%s); reading instead", error)
            return None
        async for units in corpus.batches():
            for text in _document_texts(units) or []:
                table = parse_table(text)
                if table is not None:
                    run.add(table.rows)
        answer = run.answer()
        return CountResult(
            plan=plan.model_copy(update={"list_items": plan.list_items or table_query.list_rows}),
            method="table",
            total=answer.total,
            units=answer.rows,
            groups=answer.groups,
            evidence=[render_row(summary.columns, row) for row in answer.matched],
            target_names=answer.target_values,
            items_listed=answer.matched_count,
            llm_calls=1,
            table_query=table_query,
        )

    async def count_by_reading(self, plan: CountPlan, corpus: Corpus) -> CountResult:
        """List matching items from every shard in parallel, batch by batch, then dedupe,
        merge and tally."""
        if not corpus.chars and corpus.kept is None:
            corpus.chars = sum(corpus.sizes.values())  # no pass yet: the stored size
        tokens_estimate = corpus.chars // 4
        if tokens_estimate > self.max_reading_tokens:
            raise BroadLimitError(
                f"BROAD: this question needs every chunk read by an LLM, about "
                f"{tokens_estimate:,} tokens, more than max_reading_tokens "
                f"({self.max_reading_tokens:,}). Ask a narrower question (one a table or "
                "log can answer by code), count a smaller dataset, or raise "
                "max_reading_tokens in retriever_specific_config."
            )
        semaphore = asyncio.Semaphore(self.max_parallel_calls)

        async def read_once(shard: list[Unit]) -> ShardItems:
            async with semaphore:
                return await self.extract(plan, shard)

        async def read(shard: list[Unit]) -> tuple[str, ShardItems]:
            return shard[0].id, await read_once(shard)

        read_shards: list[tuple[str, ShardItems]] = []
        passages: list[str] = []
        shard_count = tokens_read = units_read = 0
        async for units in corpus.batches():
            shards, tokens = self.pack_shards(units)
            shard_count += len(shards)
            tokens_read += tokens
            units_read += len(units)
            read_shards += await asyncio.gather(*map(read, shards))
            if plan.target and len(passages) < BROAD_TARGET_PASSAGES:
                passages += _passages_naming(plan.target, units)
        if not units_read:
            raise NoDataError("No data found in the system, please add data first.")
        passages = passages[:BROAD_TARGET_PASSAGES]
        logger.info(
            "BROAD read %d pieces; items per piece: %s",
            shard_count,
            [len(shard_items.items) for _, shard_items in read_shards],
        )

        # The model may list one occurrence twice. Without a key the quote is the
        # only identity, and only within one unit: identical quotes from different
        # units (table rows repeating a value) are different items. With a key,
        # fifty records in one piece may share a templated sentence ("The run
        # failed") and are still fifty items; the key tells them apart.
        seen: set[tuple[str, int, str | None, str]] = set()
        items: list[ExtractedItem] = []
        for shard_id, shard_items in read_shards:
            for item in shard_items.items:
                # A keyed item listed twice in one unit is one item. An unkeyed entry
                # is what the model said it is, one per occurrence: a sponsor read or an
                # interruption is written with the same words each time, and two of
                # them in one unit are two items, not a repeat.
                marker = (shard_id, item.unit, item.group, item.key) if item.key else None
                if marker is None or marker not in seen:
                    if marker is not None:
                        seen.add(marker)
                    if item.undone and not plan.reversible:
                        # "How many orders were cancelled" counts the cancellations
                        # themselves; only a plan that counts items in effect subtracts.
                        item = item.model_copy(update={"undone": False})
                    items.append(item)
        aliases = _alias_groups(
            [names for _, shard_items in read_shards for names in shard_items.aliases]
        )
        items = _one_entry_per_group(items, relation=plan.relation)
        items = _drop_attribute_named_groups(items, plan.group_by)

        canonical = await self.merge_name_variants(plan, items, aliases)
        if plan.relation and canonical:
            # For a relation the key is the other participant, a name: spell it as
            # the group it would be, so "Art" and "Arthur Bennett" are one connection.
            for item in items:
                if item.key:
                    item.key = canonical.get(item.key, item.key)

        unkeyed = 0
        entries_read = len(items)
        keyed = sum(1 for item in items if item.key)
        if plan.dedup_key and keyed * 2 >= len(items):
            # The key is how these items are written. An entry without it is, as a rule,
            # a record continued past a chunk boundary, whose identity was read where it
            # began: it is left out, and the answer says how many were (the most the
            # count could be under by).
            unkeyed = sum(1 for item in items if not item.key and item.undone == plan.counts_ended)
            items = [
                item
                for item in self.dedup(
                    items, by_group=plan.relation, counts_ended=plan.counts_ended
                )
                if item.key
            ]
        else:
            # No key planned, or one the text does not write (most entries lack it): every
            # entry is its own occurrence.
            # Without an identity an undone entry cannot name what it undoes.
            items = [item for item in items if item.undone == plan.counts_ended]

        # Each item counts 1, or its stated amount when the plan sums a measure.
        def weight(item: ExtractedItem) -> float:
            return (item.amount or 0) if plan.measure else 1

        group_totals: dict[str, float] = {}
        for item in items:
            if item.group:
                group_totals[item.group] = group_totals.get(item.group, 0) + weight(item)
        groups = sorted(group_totals.items(), key=lambda pair: -pair[1])
        items_listed = len(items)
        target_names: list[str] = []
        if plan.target:
            # The question's name is matched against the names actually read, so a
            # nickname or partial name finds its person and an unknown name counts zero.
            target_names = await self.match_target(
                plan.target, [name for name, _ in groups], aliases, canonical or {}, passages
            )
            items = [item for item in items if item.group in target_names]
        denominator = 0
        if plan.ratio_condition:
            # "What share of tickets were escalated": every ticket was listed, each
            # marked as meeting the condition or not; both counts are code's.
            denominator = len(items)
            items = [item for item in items if item.matches]
        if plan.distinct and not plan.target and _groups_are_keys(items):
            # Grouped by the item's own identifier ("different bills" by bill number): one
            # bill written "H.R. 4418" in one place and "HR4418" in another is still one,
            # which the normalized key knows and the raw group spelling does not.
            # Groups are the spellings being reconciled, so labels are adopted across them.
            ungrouped = [
                (_normalize_key(i.key), i.model_copy(update={"group": None}))
                for i in items
                if i.key
            ]
            total: float = len({key for key, _ in _adopt_labels(ungrouped)})
        elif plan.distinct and not plan.target:
            total = len(groups)
        elif plan.relation and not plan.target:
            # "How many connections were removed": each relation once, not once per
            # participant; the per-participant tally stays for "who has the most".
            total = len({frozenset((item.group, item.key)) for item in items if item.key})
        elif plan.measure and plan.average:
            amounts = [item.amount for item in items if item.amount is not None]
            total = sum(amounts) / len(amounts) if amounts else 0
        else:
            total = sum(weight(item) for item in items)
        return CountResult(
            plan=plan,
            method="reading",
            total=total,
            units=units_read,
            denominator=denominator,
            groups=groups,
            items_listed=items_listed,
            entries_read=entries_read,
            evidence=[
                f"{item.key}: {item.evidence}" if item.key else item.evidence for item in items
            ],
            amounts_missing=(
                sum(1 for item in items if item.amount is None) if plan.measure else 0
            ),
            names_merged=canonical is not None,
            target_names=target_names,
            unkeyed=unkeyed,
            llm_calls=shard_count,
            tokens_read=tokens_read,
        )

    # --- reading helpers ------------------------------------------------------------

    @staticmethod
    def dedup(
        items: list[ExtractedItem], by_group: bool = False, counts_ended: bool = False
    ) -> list[ExtractedItem]:
        """One item per (group, key), in first-seen order, minus the items undone later
        (or, with ``counts_ended``, only the items whose latest entry ended them).

        The key identifies the item (the planner defines it as unique across the
        corpus); the group is the value it is counted under, and an item with several
        values (a paper by three authors) is one item of each. Groups are merged
        spellings by now. An entry with a key but no group is a recap of an item
        already counted under some group. An entry without a key counts once (it
        cannot be checked for repeats). An entry marked undone removes the item it
        names instead of counting.
        """

        # A relation's key is the other participant, a name; otherwise an identifier.
        normalize = _loose_name if by_group else _normalize_key

        keyed = [(normalize(item.key), item) for item in items if item.key]
        unkeyed = [item for item in items if not item.key and item.undone == counts_ended]
        if not by_group:
            keyed = _adopt_labels(keyed)
        # An item's state is decided by its latest entry: opened, resolved, reopened
        # is open. Entries are ordered by the date they carry, then by reading order
        # (documents ingested as separate files have no order of their own). An
        # entry with no group is a recap of the item under some group and never
        # changes its state on its own.
        first: dict[tuple[str, str], ExtractedItem] = {}
        states: dict[tuple[str, str], list[tuple[tuple[str, int], bool]]] = {}
        # A ticket's escalation is its own email: the ratio flag holds if any entry
        # of the item says so, and an amount comes from whichever entry states it.
        matched: dict[tuple[str, str], bool] = {}
        amounts: dict[tuple[str, str], float] = {}
        # "Ticket 7 was resolved" with no group ends ticket 7 under whichever group.
        ended_by_key: dict[str, list[tuple[str, int]]] = {}
        # Only when items are grouped is an entry without a group a recap; ungrouped,
        # every entry is the item's own.
        grouped = any(item.group for _, item in keyed)
        for position, (key, item) in enumerate(keyed):
            stamp = (item.when or "", position)
            if grouped and item.group is None and item.undone:
                ended_by_key.setdefault(key, []).append(stamp)
                continue
            identity = (item.group or "", key)
            first.setdefault(identity, item)
            states.setdefault(identity, []).append((stamp, item.undone))
            matched[identity] = matched.get(identity, False) or bool(item.matches)
            if item.amount is not None:
                amounts.setdefault(identity, item.amount)
        keys_with_group = {identity[1] for identity in first if identity[0]}
        counted = []
        for identity, entries in states.items():
            entries += [(stamp, True) for stamp in ended_by_key.get(identity[1], [])]
            _, undone = max(entries)
            if undone != counts_ended or (not identity[0] and identity[1] in keys_with_group):
                continue
            item = first[identity]
            if item.matches is not None or matched[identity]:
                item = item.model_copy(update={"matches": matched[identity]})
            if item.amount is None and identity in amounts:
                item = item.model_copy(update={"amount": amounts[identity]})
            counted.append(item)
        return [*counted, *unkeyed]

    def pack_shards(self, units: list[Unit]) -> tuple[list[list[Unit]], int]:
        """Pack whole units into calls of at most ``shard_tokens``; a unit larger than
        that is a call by itself. Returns (shards, tokens)."""
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
        condition = plan.condition or "none"
        if plan.reversible and plan.condition:
            # The planner may restate the state ("currently open, not resolved");
            # applied per entry that would skip the very entries that end an item.
            condition += (
                " — this selects which items are of interest, never their current "
                "state: list EVERY state entry (opened, resolved, reopened, removed) of "
                "such items; code decides the state from the latest entry"
            )
        spec = (
            f"Item: {plan.item}\n"
            f"Condition: {condition}\n"
            f"Identity attribute (key): {plan.dedup_key or 'none'}\n"
            f"Grouping attribute (group): {plan.group_by or 'none'}"
            + (
                f" — write its value for this item (a name, a place, a category) exactly as "
                f'written, never the word "{plan.group_by}" itself and never a description '
                f"of the same thing (a person's role, a place's region): the name\n"
                if plan.group_by
                else "\n"
            )
            + f"Amount to report (amount): {plan.measure or 'none'}\n"
            f"Ratio condition (matches): {plan.ratio_condition or 'none'}"
            + (
                " — list EVERY item and set matches = true or false for each"
                if plan.ratio_condition
                else ""
            )
            + "\nUndone entries: "
            + (
                "the item's state changes over time; list every entry that records a "
                "state, with its date in when: undone = true for an entry that ends it "
                "(resolved, closed, removed, cancelled after it was placed, returned), "
                "undone = false for one that starts or restarts it (opened, reopened, "
                "connected, placed). Code keeps the latest entry per item."
                if plan.reversible
                else "not applicable; never set undone"
            )
            + "\nRelation: "
            + (
                "the item relates two or more things of one kind; list it ONCE, with EVERY "
                "participant in groups (a paper by four authors: all four names), whatever "
                "their number. One that was only requested, proposed or declined is not "
                "the relation."
                if plan.relation
                else "no"
            )
        )
        # Unit markers let items be traced back; units are read whole. A document's
        # first line is shown once per call; the end of the previous chunk is shown
        # before a chunk read without it.
        blocks: list[str] = []
        shown: set[str] = set()
        in_call = {unit.id for unit in shard}
        for index, unit in enumerate(shard):
            block = f"[unit {index}]\n{unit.text}"
            if unit.previous_text and unit.previous_id not in in_call:
                context = _tail(unit.previous_text)
                block = f"[end of the previous chunk — context only]\n{context}\n\n{block}"
            if unit.preamble and unit.preamble not in shown:
                shown.add(unit.preamble)
                block = f"[document start — context only]\n{unit.preamble}\n\n{block}"
            blocks.append(block)
        return await self._ask(
            f"{spec}\n\nSECTION:\n" + "\n\n".join(blocks),
            "broad_extract.txt",
            ShardItems,
        )

    async def merge_name_variants(
        self, plan: CountPlan, items: list[ExtractedItem], aliases: list[list[str]]
    ) -> dict[str, str] | None:
        """Rewrite every group value to one canonical spelling, in place.

        Returns the spelling each name was mapped to, or None when there were too
        many distinct names to merge.
        """
        # Only group values are names. A dedup key is an identifier (a number, a
        # code, a title): merging "similar" identifiers would join different items.
        names = {item.group for item in items if item.group}
        if plan.relation:
            names |= {item.key for item in items if item.key}
        if len(names) > BROAD_MAX_ALIAS_NAMES:
            return None
        if len(names) < 2:
            return {name: name for name in names}

        # Only names that were read reach the model as stated aliases: a string that
        # is not a read name cannot change a count, and one that is a whole sentence
        # ("Pavel Horák is Pav") would make the model merge whatever else stands
        # next to it.
        stated = _stated_aliases(
            [group for group in _match_stated(aliases, names) if len(set(group)) > 1]
        )
        text_input = "\n".join(sorted(names))
        if stated:
            text_input += "\n\nStated in the text to be the same:\n" + "\n".join(stated)
        result = await self._ask(
            text_input,
            "broad_merge_names.txt",
            NameGroups,
        )
        canonical = {name: name for name in names}
        used = Counter(item.group for item in items if item.group)
        if plan.relation:
            used.update(item.key for item in items if item.key)
        # What the text itself declares equal ("Arthur Bennett (usually called Art)")
        # is applied by code; the model's groups add what it is sure of on top.
        for group in [*_match_stated(aliases, names), *result.groups]:
            members = {canonical[variant] for variant in group if variant in canonical}
            if len(members) > 1:
                head = _canonical_spelling(sorted(members), used)
                used[head] = sum(used[member] for member in members)
                for variant, current in canonical.items():
                    if current in members:
                        canonical[variant] = head
        logger.info(
            "BROAD name merge: %d names, %d merged into another spelling",
            len(names),
            sum(1 for name, target in canonical.items() if name != target),
        )
        for item in items:
            if item.group:
                item.group = canonical[item.group]
        return canonical

    async def match_target(
        self,
        target: str,
        names: list[str],
        aliases: list[list[str]],
        canonical: dict[str, str],
        passages: list[str] | None = None,
    ) -> list[str]:
        """The names read from the corpus that are the person or thing ``target`` names.

        Matched by code when the question's name is a spelling that was read, or
        one the text declares equal to it; a model call only resolves the rest
        (a nickname, a translation, an id declared in another document), against
        the names actually present and the passages that write the name out.
        """
        if not names:
            return []
        wanted = {_loose_name(target)}
        for group in aliases:
            if any(_loose_name(name) in wanted for name in group):
                wanted.update(_loose_name(name) for name in group)
        spelling_of = {_loose_name(name): name for name in names}
        spelling_of.update({_loose_name(variant): head for variant, head in canonical.items()})
        by_spelling = sorted({spelling_of[w] for w in wanted if spelling_of.get(w) in names})
        if not by_spelling:
            # "John Jay" when the text writes "JAY": a corpus name that is one whole word of
            # the question's name is it, when it is the only such name.
            words = {_loose_name(word) for word in target.split() if len(_loose_name(word)) > 2}
            by_word = [name for name in names if _loose_name(name) in words]
            if len(by_word) == 1:
                by_spelling = by_word
        others = [name for name in names if name not in by_spelling]
        if not others:
            return by_spelling

        # The model adds what spelling cannot see (a translation, a code, a nickname
        # the text never explains), anchored on the spellings already known to match.
        text_input = f"Name in the question: {target}\n"
        if by_spelling:
            text_input += "Corpus names already known to be it: " + ", ".join(by_spelling) + "\n"
        text_input += "\nNames in the corpus:\n" + "\n".join(others)
        stated = _stated_aliases(aliases)
        if stated:
            text_input += "\n\nStated in the text to be the same:\n" + "\n".join(stated)
        if passages:
            text_input += "\n\nPassages where the question's name is written:\n" + "\n".join(
                f"- {passage}" for passage in passages
            )
        result = await self._ask(
            text_input,
            "broad_match_target.txt",
            TargetMatch,
        )
        matched = by_spelling + [name for name in others if name in set(result.names)]
        logger.info(
            "BROAD target %r matched %s (%d by spelling)", target, matched, len(by_spelling)
        )
        return matched

    # --- context / completion -------------------------------------------------------

    async def get_context_from_objects(self, query: str, retrieved_objects: CountResult) -> str:
        result = retrieved_objects
        plan = result.plan
        if result.method == "unsupported":
            return (
                "This question cannot be answered by counting: "
                f"{plan.unsupported} Say so plainly, and do not give a number."
            )
        if result.method == "table" and result.table_query:
            return _table_context(result)
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
        if plan.reversible:
            state = "ended (resolved, closed, removed)" if plan.counts_ended else "still in effect"
            lines.append(f"Counted: the items whose latest recorded state is {state}")
        if result.entries_read > result.items_listed:
            # Said out loud so a key that folds different items together is visible
            # in the answer rather than silently shrinking the count.
            lines.append(
                f"({result.entries_read} entries were listed while reading; "
                f"{result.items_listed} remain after removing repeated mentions)"
            )
        if result.unkeyed:
            lines.append(
                f"({result.unkeyed} entries whose {plan.dedup_key} could not be read were left "
                "out as repeats of counted items: the total may be under by up to that many)"
            )
        lines.append(f"TOTAL: {_number(result.total)}")
        if plan.ratio_condition:
            share = 100 * result.total / result.denominator if result.denominator else 0
            lines.append(
                f"  = the items meeting the condition, out of {result.denominator} items in all: "
                f"{share:.1f}% (both counted by code)"
            )
        if plan.measure and plan.average:
            lines.append(f"  = the average {plan.measure} over the listed items")
        elif plan.measure:
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
        self,
        query: str,
        retrieved_objects: Any,
        context: Any | None = None,
        effective_query: str | None = None,
        turn_preparation=None,
    ) -> list[Any]:
        """The LLM phrases the count; code appends how it was obtained and any requested
        list, so neither depends on the answer model."""
        completions = await super().get_completion_from_context(
            query,
            retrieved_objects,
            context=context,
            effective_query=effective_query,
            turn_preparation=turn_preparation,
        )
        tail = []
        note = _provenance_note(retrieved_objects)
        if note:
            tail.append(note)
        entries = self.listing(retrieved_objects)
        if retrieved_objects.plan.list_items and entries:
            # A table's rows are rendered only up to the listing limit; all were matched.
            table = retrieved_objects.method == "table"
            count = retrieved_objects.items_listed if table else len(entries)
            block = [f"Full list ({count}):"]
            block += [f"- {entry}" for entry in entries[:BROAD_MAX_LISTED]]
            if count > BROAD_MAX_LISTED:
                block.append(f"... and {count - BROAD_MAX_LISTED} more")
            tail.append("\n".join(block))
        if not tail:
            return completions
        suffix = "\n\n".join(tail)
        return [
            f"{completion.rstrip()}\n\n{suffix}" if isinstance(completion, str) else completion
            for completion in completions
        ]

    def extract_context_object_ids(self, retrieved_objects: Any) -> dict[str, list[str]] | None:
        return None

    def get_context_evidence(self, retrieved_objects: Any, dataset_id: Any = None):
        return None

    async def append_references(self, completions: list[Any], retrieved_objects: Any) -> list[Any]:
        return completions
