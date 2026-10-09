"""Show the model each subject's competing facts and ask it to judge them.

One call shows a handful of subjects with their own facts, the facts of anything
competing with them for a value, and the conflicts already stored about them.
Everything in a call is labelled, and only labels this call rendered may come
back — ``validate_review_output`` resolves them and rejects the rest.
"""

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.utils import get_llm_token_ceiling, get_llm_tokenizer
from cognee.modules.graph.utils.fact_conflicts import effective_date_display
from cognee.shared.logging_utils import get_logger

from .facts import FactIndex, cited_fact_ids, endpoints, fact_priority, is_statement
from .models import ReviewBatch, ReviewScope
from .schema import ReviewOutput
from .trim_facts_to_budget import TokenBudget, fit_to_budget
from .validate_review import validate_review_output

ENTITIES_PER_CALL = 10
INPUT_TOKEN_SHARE = 0.5
SPLIT_RETRIES = 1

PROMPT_FILE = "review_conflicts.txt"

logger = get_logger("review_conflicts")


def _review_budget(system_prompt: str) -> TokenBudget:
    """Every call pays the prompt and the response schema before its first fact line."""
    tokenizer = get_llm_tokenizer()
    overhead = tokenizer.count_tokens(system_prompt) + tokenizer.count_tokens(
        json.dumps(ReviewOutput.model_json_schema(), sort_keys=True)
    )
    return TokenBudget(tokenizer, overhead, int(get_llm_token_ceiling() * INPUT_TOKEN_SHARE))


# --- the header block --------------------------------------------------------


def _conflict_participants(conflict: dict) -> list[str]:
    """The subject a stored conflict is about, plus each value it names."""
    return [str(conflict["about_id"]), *conflict.get("values", [])]


@dataclass(frozen=True)
class SubjectHeaders:
    """One call's header block: its stored-conflict labels and each subject's lines."""

    conflict_labels: dict[str, str]  # `f1` -> conflict id, only for conflicts a subject owns
    lines_by_subject: dict[str, list[str]]

    def as_text(self) -> str:
        return "\n\n".join("\n".join(lines) for lines in self.lines_by_subject.values())


def _subject_headers(scope: ReviewScope, subject_ids: list[str]) -> SubjectHeaders:
    """Name each subject with its types, then the stored conflicts it takes part in."""
    subjects = set(subject_ids)
    related = {
        conflict_id: conflict
        for conflict_id, conflict in scope.conflicts.items()
        if subjects.intersection(_conflict_participants(conflict))
    }
    # Only the owner of a conflict may replace or drop it, so only those get a label.
    owned = sorted(
        conflict_id
        for conflict_id, conflict in related.items()
        if str(conflict["about_id"]) in subjects
    )
    conflict_labels = {f"f{number}": value for number, value in enumerate(owned, 1)}
    label_of_conflict = {value: label for label, value in conflict_labels.items()}

    lines_by_subject = {}
    for number, subject in enumerate(subject_ids, 1):
        properties = scope.entities[subject]
        types = ", ".join(properties.get("types", []))
        lines = [f"Subject [n{number}] {properties['name']} (types: {types})"]
        for conflict_id, conflict in related.items():
            if subject not in _conflict_participants(conflict):
                continue
            owner = "subject" if subject == str(conflict["about_id"]) else "value"
            label = (
                f" [{label_of_conflict[conflict_id]}]" if conflict_id in label_of_conflict else ""
            )
            pending = (
                "; pending write" if owner == "subject" and conflict.get("review_pending") else ""
            )
            lines.append(f"Existing{label} {conflict['attribute']} ({owner}{pending})")
        lines_by_subject[subject] = lines
    return SubjectHeaders(conflict_labels, lines_by_subject)


def _header_block_text(scope: ReviewScope, subject_ids: list[str]) -> str:
    """Size a prospective call without building its fact context."""
    return _subject_headers(scope, subject_ids).as_text()


# --- the rendered call -------------------------------------------------------


def _numbered_labels(prefix: str, node_ids: list[str]) -> dict[str, str]:
    """`n1`, `n2`, … in the given order: the prompt names every node and fact by label."""
    return {f"{prefix}{number}": node_id for number, node_id in enumerate(node_ids, 1)}


def _cited_source_text(source: dict) -> str:
    """Name the document and, where known, the date it states the fact for."""
    date = effective_date_display(source.get("effective_date"))
    document = source.get("document") or source.get("data_id") or "unnamed document"
    return f"{document} ({date})" if date else str(document)


@dataclass(frozen=True)
class RenderedCall:
    """The exact text one call sent, and the labels it means."""

    text: str
    entities: dict[str, str]
    facts: dict[str, str]
    conflicts: dict[str, str]
    fact_lines: dict[str, list[str]]


class ReviewCall:
    """Labels and shown facts belong to one call, including a split retry."""

    def __init__(
        self,
        scope: ReviewScope,
        subject_ids: list[str],
        fact_index: FactIndex | None = None,
    ):
        self.scope = scope
        self.subject_ids = subject_ids
        self.fact_index = fact_index or FactIndex(scope.facts)
        shown = {subject: self._facts_shown_for(subject) for subject in subject_ids}
        self.priority = self._fact_priorities(set().union(*shown.values()))
        self.shown_facts = {
            subject: sorted(fact_ids, key=lambda fact_id: self.rank(subject, fact_id))
            for subject, fact_ids in shown.items()
        }

    def _facts_shown_for(self, subject: str) -> set[str]:
        """A subject's own facts, plus every fact touching an entity that competes with it."""
        own = self.fact_index.facts_touching(subject)
        rivals = {
            endpoint
            for fact_id in self.fact_index.competing_with(own)
            for endpoint in endpoints(self.scope.facts[fact_id])
        }
        return own | set().union(*(self.fact_index.facts_touching(rival) for rival in rivals))

    def _fact_priorities(self, shown: set[str]) -> dict[str, tuple]:
        cited = cited_fact_ids(self.scope, shown)
        return {
            fact_id: fact_priority(
                self.scope.facts[fact_id],
                self.scope.since,
                cited,
                self.fact_index.groups_by_fact.keys(),
            )
            for fact_id in shown
        }

    def rank(self, subject: str, fact_id: str) -> tuple:
        """Only a subject's own facts may be labelled for it, so context sorts last."""
        fact = self.scope.facts[fact_id]
        return (0 if subject in endpoints(fact) else 1, *self.priority[fact_id])

    # --- rendering ---

    def render(self) -> RenderedCall:
        shown = set().union(*self.shown_facts.values())
        entities = self._entity_labels(shown)
        facts = self._fact_labels(shown)
        headers = _subject_headers(self.scope, self.subject_ids)
        label_of_entity = {node_id: label for label, node_id in entities.items()}
        label_of_fact = {fact_id: label for label, fact_id in facts.items()}
        fact_lines = {
            subject: [
                self._fact_line(fact_id, subject, label_of_entity, label_of_fact)
                for fact_id in fact_ids
            ]
            for subject, fact_ids in self.shown_facts.items()
        }
        sections = [
            "\n".join(
                [
                    *headers.lines_by_subject[subject],
                    *fact_lines[subject],
                    *self._competing_lines(subject, label_of_fact),
                ]
            )
            for subject in self.subject_ids
        ]
        return RenderedCall(
            "\n\n".join(sections + self._context_names(entities)),
            entities,
            facts,
            headers.conflict_labels,
            fact_lines,
        )

    def _entity_labels(self, shown: set[str]) -> dict[str, str]:
        """Subjects keep the order they were batched in; other endpoints follow, sorted."""
        endpoint_ids = {
            endpoint
            for fact_id in shown
            if not is_statement(self.scope.facts[fact_id])
            for endpoint in endpoints(self.scope.facts[fact_id])
        }
        return _numbered_labels(
            "n", self.subject_ids + sorted(endpoint_ids - set(self.subject_ids))
        )

    def _fact_labels(self, shown: set[str]) -> dict[str, str]:
        """`[s#]` is a chunk statement about a subject, `[r#]` a claim between two entities."""
        statements = sorted(f for f in shown if is_statement(self.scope.facts[f]))
        relationships = sorted(f for f in shown if not is_statement(self.scope.facts[f]))
        return {**_numbered_labels("s", statements), **_numbered_labels("r", relationships)}

    def _fact_line(
        self,
        fact_id: str,
        subject: str,
        label_of_entity: dict[str, str],
        label_of_fact: dict[str, str],
    ) -> str:
        fact = self.scope.facts[fact_id]
        relation = (
            "statement"
            if is_statement(fact)
            else f"{label_of_entity[fact['source']]} --{fact['relationship']}--> "
            f"{label_of_entity[fact['target']]}"
        )
        context = "" if subject in endpoints(fact) else "context; "
        statement = fact["properties"].get("edge_text") or fact["relationship"]
        sources = "; ".join(_cited_source_text(source) for source in fact["sources"])
        return (
            f"[{label_of_fact[fact_id]}] {context}{relation} | "
            f"{statement} | {sources or 'undated source'}"
        )

    def _competing_lines(self, subject: str, label_of_fact: dict[str, str]) -> list[str]:
        """Groups of this subject's facts the LLM should compare against each other."""
        section = set(self.shown_facts[subject])
        lines = []
        for index in sorted(self.fact_index.group_indices(section)):
            # A label this section does not show cannot be compared here:
            # validate_review_output rejects it and the whole batch is discarded.
            labels = sorted(
                label_of_fact[fact_id] for fact_id in self.fact_index.groups[index] & section
            )
            if len(labels) > 1:
                lines.append(f"Competing relationships: {', '.join(labels)}")
        return lines

    def _context_names(self, entities: dict[str, str]) -> list[str]:
        """Non-subject entities are named once at the end; their labels appear in fact lines."""
        subjects = set(self.subject_ids)
        return [
            f"[{label}] {self.scope.nodes.get(node_id, {}).get('name', node_id)}"
            for label, node_id in entities.items()
            if node_id not in subjects
        ]


# --- one run over a dataset's subjects ---------------------------------------


@dataclass
class ReviewRun:
    """What every call in one run shares: the scope, its index, the prompt and the budget."""

    scope: ReviewScope
    fact_index: FactIndex
    system_prompt: str
    budget: TokenBudget
    unreviewed_entity_ids: list[str] = field(default_factory=list)

    def plan_calls(self) -> Iterator[list[str]]:
        """Group subjects into calls: at most ENTITIES_PER_CALL, headers must fit the budget."""
        subjects: list[str] = []
        for subject in self.scope.entities:
            if not self.budget.fits(_header_block_text(self.scope, [subject])):
                self.unreviewed_entity_ids.append(subject)
                logger.warning("Fact review header exceeds input budget for entity %s", subject)
                continue
            if subjects and (
                len(subjects) == ENTITIES_PER_CALL
                or not self.budget.fits(_header_block_text(self.scope, [*subjects, subject]))
            ):
                yield subjects
                subjects = []
            subjects.append(subject)
        if subjects:
            yield subjects

    async def review(self, subjects: list[str], retries: int) -> AsyncIterator[ReviewBatch]:
        """One LLM call, or — when it fails — two calls over half the subjects each."""
        call = ReviewCall(self.scope, subjects, self.fact_index)
        # Outside the try: a call whose headers alone exceed the budget cannot be
        # salvaged by splitting, and the error belongs to the run.
        rendered = fit_to_budget(call, self.budget)
        try:
            batch = await self._ask_llm(call, rendered)
        except LLMPaymentRequiredError:
            raise
        except Exception as error:
            if retries and len(subjects) > 1:
                async for batch in self._split_in_half(subjects, retries):
                    yield batch
            else:
                self.unreviewed_entity_ids.extend(subjects)
                logger.warning(
                    "Fact review failed for entities %s: %s", subjects, error, exc_info=True
                )
            return
        self.unreviewed_entity_ids.extend(batch.unreviewed_entity_ids)
        yield batch

    async def _ask_llm(self, call: ReviewCall, rendered: RenderedCall) -> ReviewBatch:
        output = await LLMGateway.acreate_structured_output(
            text_input=rendered.text,
            system_prompt=self.system_prompt,
            response_model=ReviewOutput,
        )
        return validate_review_output(ReviewOutput.model_validate(output), call, rendered)

    async def _split_in_half(self, subjects: list[str], retries: int) -> AsyncIterator[ReviewBatch]:
        """Half the subjects may still answer where the whole batch did not."""
        midpoint = len(subjects) // 2
        for half in (subjects[:midpoint], subjects[midpoint:]):
            async for batch in self.review(half, retries - 1):
                yield batch


async def review_entities(scope: ReviewScope) -> AsyncIterator[ReviewBatch]:
    """Yield independent writes; storage errors cannot enter the LLM retry handler.

    The calls run one at a time on purpose: the writer mutates this ``ReviewScope``
    between yields, and the next call reads those marks, citations and conflicts.
    Nothing in the suite pins that sequencing — the test named
    ``test_reused_index_reads_support_changed_by_previous_batch`` only pins that
    the state is re-read rather than cached alongside the immutable fact index.
    """
    system_prompt = render_prompt(PROMPT_FILE, {})
    run = ReviewRun(
        scope=scope,
        fact_index=FactIndex(scope.facts),
        system_prompt=system_prompt,
        budget=_review_budget(system_prompt),
    )
    if scope.drop_conflict_ids:
        yield ReviewBatch(scope, dropped_conflict_ids=list(scope.drop_conflict_ids))
    for subjects in run.plan_calls():
        async for batch in run.review(subjects, SPLIT_RETRIES):
            yield batch
    yield ReviewBatch(scope, final=True, unreviewed_entity_ids=run.unreviewed_entity_ids)
