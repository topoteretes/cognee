"""Collapse a recall window into distinct asks, then dedup them per partition.

Phase 1 steps 2, 5 and 6 of the recall-coverage spec.

**The partition key is ``(user_id, dataset_id)``.** Two questions merge only when
their text is similar AND they came from the same user AND they were asked
against the same dataset, because a question row *is*
``(user_id, dataset_id, canonical text)``: the same text from two teammates, or
against two brains, is two rows and two independent coverage answers. Rows whose
``dataset_id`` is NULL — a search that spanned several datasets, or a curated
question — form their own partition rather than being special-cased.

Dedup stays non-quadratic because the input is capped at ``max_questions``, not
because of an index: there is no ANN index anywhere in this repo and no LSH here.
Within a partition it is one ``X @ X.T`` and greedy single-link grouping.

Nothing in this module calls an LLM or touches the database.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional, Sequence
from uuid import UUID, uuid4

import numpy as np

from cognee.modules.recall_coverage.types import QuestionSource
from cognee.modules.search.operations.get_queries import QueryWindowRow


@dataclass
class Ask:
    """One distinct ask against one ``(user_id, dataset_id)`` partition.

    Produced by :func:`collapse_asks` from observed ``queries`` rows, and
    constructed directly for curated questions (``source = "curated"``,
    ``first_seen = None``, no ``query_ids``), which enter the same dedup as
    everything else.
    """

    text: str
    user_id: UUID
    dataset_id: Optional[UUID] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    query_type: Optional[str] = None
    query_ids: list[UUID] = field(default_factory=list)
    source: str = QuestionSource.OBSERVED.value
    curated_question_id: Optional[UUID] = None

    # Shared by the asks that are one search fanned across several datasets. See
    # the fan-out comment in ``collapse_asks``: provenance only, never counted.
    fanout_group_id: Optional[UUID] = None

    # How many extra rows the retry cooldown swallowed into this ask.
    retry_collapsed_count: int = 0

    @property
    def is_observed(self) -> bool:
        """True when an agent actually asked this, as opposed to a human adding it."""
        return self.source == QuestionSource.OBSERVED.value

    @property
    def partition_key(self) -> tuple[UUID, Optional[UUID]]:
        return (self.user_id, self.dataset_id)


@dataclass(frozen=True)
class CollapseResult:
    """What one window collapsed into, plus the counters the run row reports."""

    asks: list[Ask]
    # Raw ``queries`` rows in the window, before any collapsing.
    recall_row_count: int
    # Asks after the retry cooldown, before ``max_questions`` truncation.
    distinct_ask_count: int
    # Rows the retry cooldown swallowed. Reported so a reader can see how much
    # of the traffic was one agent looping.
    collapsed_retry_count: int
    # Asks dropped by the ``max_questions`` truncation (oldest first).
    dropped_ask_count: int
    # Distinct fan-out groups seen. Provenance only; no aggregate uses it.
    fanout_group_count: int


@dataclass
class DedupedQuestion:
    """One question row: a cluster of similar asks inside a single partition."""

    text: str
    user_id: UUID
    dataset_id: Optional[UUID]
    source: str
    was_asked: bool
    # Distinct *asks* in the cluster, never rows, and never counting curated
    # members — a human adding a question is not demand for it.
    occurrence_count: int
    first_asked_at: Optional[datetime]
    last_asked_at: Optional[datetime]
    curated_question_id: Optional[UUID]
    # Row of the embedding matrix holding this question's canonical vector, so
    # later phases (question groups, topic assignment) reuse it instead of
    # re-embedding.
    canonical_index: int
    # Indices into the ``asks`` list this cluster was built from.
    ask_indices: list[int]
    query_ids: list[UUID]
    # Stamped by :func:`assign_question_groups`, after every partition is done.
    question_group_id: Optional[UUID] = None


@dataclass(frozen=True)
class DedupResult:
    questions: list[DedupedQuestion]
    # Pairwise similarities actually considered. Bounded by ``max_questions``
    # (see the module docstring), which is what the caller asserts against.
    comparison_count: int
    partition_count: int


def collapse_text_key(text: str) -> str:
    """Identity for the exact-match collapse rules below.

    Casefolded and whitespace-collapsed, so ``"Where are the runbooks?"`` and
    ``"where are  the runbooks? "`` are the same ask. Anything looser than exact
    is dedup's job, not collapse's — collapse runs before any embedding exists.
    """
    return " ".join((text or "").split()).casefold()


def _seconds_between(later: datetime, earlier: datetime) -> float:
    return abs((later - earlier).total_seconds())


def collapse_asks(
    rows: Iterable[QueryWindowRow],
    *,
    fanout_window_seconds: int,
    retry_cooldown_seconds: int,
    max_questions: int,
) -> CollapseResult:
    """Collapse ``queries`` rows into distinct asks, newest first.

    Two rules, and it matters that only one of them counts:

    **Fan-out (no counting effect).** ``log_search_history`` writes one
    ``queries`` row per ``SearchResultPayload``, i.e. one per dataset, so a
    single search over three datasets is three rows with identical text and
    ``query_type`` a few milliseconds apart. Those rows are recognised here —
    identical text + ``query_type`` + user inside ``fanout_window_seconds`` share
    a ``fanout_group_id`` — purely so the string is embedded once rather than
    three times. **There is no fan-out counting rule.** Under
    ``(user_id, dataset_id)`` partitioning that fanned search is already
    *correctly* three rows in three partitions with one ask each: each dataset
    was genuinely asked once, and each answers independently. Collapsing them
    into one ask would delete two partitions' worth of coverage.

    **Retry cooldown (this is the counting rule).** Identical text + same
    ``user_id`` + same ``dataset_id`` within ``retry_cooldown_seconds`` counts as
    **one** ask: an agent looping on a question it cannot answer is not three
    times the demand. This is a **heuristic** — a human legitimately re-asking
    the same thing two minutes later is swallowed by it, and a loop that pauses
    longer than the cooldown is not — chosen because over-counting retries would
    make every broken agent look like a popular question. It chains: each row is
    compared against the running ask's earliest occurrence, so a tight loop
    collapses entirely however long it ran. What it swallowed is reported as
    ``collapsed_retry_count`` instead of being thrown away.

    Truncation to ``max_questions`` happens last, newest first, and is what keeps
    the dedup matmul bounded.
    """
    # Defensively re-sorted: every rule below reads "within N seconds of the
    # previous occurrence", which is only the intended rule when the walk is
    # newest-first. ``get_queries`` already orders DESC; this makes a caller that
    # reorders unable to silently change the counters.
    ordered = sorted(rows, key=lambda row: row.created_at, reverse=True)

    asks: list[Ask] = []
    # (text key, user_id, dataset_id) -> index of the ask still inside its cooldown.
    open_ask: dict[tuple[str, UUID, Optional[UUID]], int] = {}
    # (text key, query_type, user_id) -> (group id, earliest row seen in the group).
    open_fanout: dict[tuple[str, Optional[str], UUID], tuple[UUID, datetime]] = {}

    collapsed_retry_count = 0

    for row in ordered:
        text_key = collapse_text_key(row.text)

        fanout_key = (text_key, row.query_type, row.user_id)
        fanout = open_fanout.get(fanout_key)
        if fanout is None:
            fanout_group_id, group_first_seen = uuid4(), row.created_at
        elif _seconds_between(fanout[1], row.created_at) <= fanout_window_seconds:
            fanout_group_id, group_first_seen = fanout[0], min(fanout[1], row.created_at)
        else:
            fanout_group_id, group_first_seen = uuid4(), row.created_at
        open_fanout[fanout_key] = (fanout_group_id, group_first_seen)

        cooldown_key = (text_key, row.user_id, row.dataset_id)
        open_index = open_ask.get(cooldown_key)
        if open_index is not None:
            ask = asks[open_index]
            if _seconds_between(ask.first_seen, row.created_at) <= retry_cooldown_seconds:
                ask.query_ids.append(row.query_id)
                ask.first_seen = min(ask.first_seen, row.created_at)
                ask.last_seen = max(ask.last_seen, row.created_at)
                ask.retry_collapsed_count += 1
                collapsed_retry_count += 1
                continue

        asks.append(
            Ask(
                text=row.text,
                user_id=row.user_id,
                dataset_id=row.dataset_id,
                first_seen=row.created_at,
                last_seen=row.created_at,
                query_type=row.query_type,
                query_ids=[row.query_id],
                source=QuestionSource.OBSERVED.value,
                fanout_group_id=fanout_group_id,
            )
        )
        open_ask[cooldown_key] = len(asks) - 1

    distinct_ask_count = len(asks)
    kept = asks[:max_questions] if max_questions is not None and max_questions >= 0 else asks

    return CollapseResult(
        asks=kept,
        recall_row_count=len(ordered),
        distinct_ask_count=distinct_ask_count,
        collapsed_retry_count=collapsed_retry_count,
        dropped_ask_count=distinct_ask_count - len(kept),
        fanout_group_count=len({ask.fanout_group_id for ask in asks}),
    )


def group_by_similarity(vectors: np.ndarray, threshold: float) -> tuple[list[list[int]], int]:
    """Greedy single-link groups of row indices with cosine similarity >= ``threshold``.

    ``vectors`` must be L2-normalized (see
    :func:`cognee.modules.recall_coverage.embedding.normalize_rows`) so that
    ``vectors @ vectors.T`` is the cosine similarity matrix. Single-link via
    union-find: A merges with B and B with C makes one group even if A and C are
    below the threshold, which is what "greedy" means here and is why the
    threshold is a demand-defining parameter rather than a free knob.

    Returns the groups in first-appearance order plus the number of pairwise
    comparisons made, which is how the caller proves the cost is bounded by
    ``max_questions`` rather than by history size.
    """
    count = int(vectors.shape[0]) if vectors.ndim == 2 else 0
    if count == 0:
        return [], 0
    if count == 1:
        return [[0]], 0

    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # Keep the earliest index as the root so group order is stable.
            if right_root < left_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    similarity = vectors @ vectors.T
    comparison_count = count * (count - 1) // 2
    for index in range(count):
        for other in range(index + 1, count):
            if similarity[index, other] >= threshold:
                union(index, other)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    return [groups[root] for root in sorted(groups)], comparison_count


def _canonical_member(asks: Sequence[Ask], members: Sequence[int]) -> int:
    """Pick the cluster's canonical ask: the curated member, else the earliest asked.

    A human wrote the curated text, so it is the better label for the cluster —
    that is the whole reason a curated question merging into real traffic keeps
    its own wording (spec section 4).
    """
    curated = [index for index in members if not asks[index].is_observed]
    if curated:
        return curated[0]

    dated = [index for index in members if asks[index].first_seen is not None]
    if not dated:
        # Only reachable for observed asks with no timestamp, which collapse never
        # produces; fall back to input order rather than comparing None to None.
        return members[0]

    return min(dated, key=lambda index: asks[index].first_seen)


def dedup_asks(
    asks: Sequence[Ask], normalized: np.ndarray, *, dedup_threshold: float
) -> DedupResult:
    """Group similar asks into question rows, partitioned by ``(user_id, dataset_id)``.

    ``normalized`` must be index-aligned with ``asks`` (row *i* is the embedding
    of ``asks[i].text``) and L2-normalized.

    Per cluster, per the spec:

    * canonical text is the curated member's if there is one, else the
      earliest-asked member's;
    * ``source`` is ``curated`` when any member is curated, and that member's
      ``curated_question_id`` is carried over;
    * ``was_asked`` is true when any member is observed;
    * ``occurrence_count`` counts distinct **asks**, never rows;
    * ``first_asked_at`` / ``last_asked_at`` are min/max over **observed**
      members only, and NULL when the cluster is purely curated.
    """
    if not asks:
        return DedupResult(questions=[], comparison_count=0, partition_count=0)

    if normalized.shape[0] != len(asks):
        raise ValueError(
            f"Embedding matrix has {normalized.shape[0]} rows for {len(asks)} asks; "
            "recall-coverage dedup requires index alignment."
        )

    partitions: dict[tuple[UUID, Optional[UUID]], list[int]] = {}
    for index, ask in enumerate(asks):
        partitions.setdefault(ask.partition_key, []).append(index)

    questions: list[DedupedQuestion] = []
    comparison_count = 0

    for indices in partitions.values():
        groups, comparisons = group_by_similarity(normalized[indices], dedup_threshold)
        comparison_count += comparisons

        for group in groups:
            members = [indices[position] for position in group]
            canonical = _canonical_member(asks, members)
            observed = [asks[index] for index in members if asks[index].is_observed]
            curated = [asks[index] for index in members if not asks[index].is_observed]

            first_seen = [ask.first_seen for ask in observed if ask.first_seen is not None]
            last_seen = [ask.last_seen for ask in observed if ask.last_seen is not None]

            questions.append(
                DedupedQuestion(
                    text=asks[canonical].text,
                    user_id=asks[canonical].user_id,
                    dataset_id=asks[canonical].dataset_id,
                    source=(
                        QuestionSource.CURATED.value if curated else QuestionSource.OBSERVED.value
                    ),
                    was_asked=bool(observed),
                    occurrence_count=len(observed),
                    first_asked_at=min(first_seen) if first_seen else None,
                    last_asked_at=max(last_seen) if last_seen else None,
                    curated_question_id=(curated[0].curated_question_id if curated else None),
                    canonical_index=canonical,
                    ask_indices=members,
                    query_ids=[query_id for index in members for query_id in asks[index].query_ids],
                )
            )

    return DedupResult(
        questions=questions,
        comparison_count=comparison_count,
        partition_count=len(partitions),
    )


def assign_question_groups(
    questions: Sequence[DedupedQuestion], normalized: np.ndarray, *, dedup_threshold: float
) -> int:
    """Stamp a shared ``question_group_id`` on matching questions across partitions.

    One similarity pass over the canonical vectors of every question row,
    regardless of partition, at the same ``dedup_threshold``. Purely so the UI can
    collapse Anna's row and Ben's row on an id instead of exact-string matching,
    which would miss them whenever two partitions settled on different canonical
    text. **No LLM calls and no effect on any score** — the rows stay separate
    rows, and every aggregate is still a mean over rows.

    Every question ends up with a group id: its own when nothing matched. Returns
    the number of distinct groups.
    """
    if not questions:
        return 0

    canonical_rows = np.asarray(
        [normalized[question.canonical_index] for question in questions], dtype=float
    )
    groups, _comparisons = group_by_similarity(canonical_rows, dedup_threshold)

    for group in groups:
        group_id = uuid4()
        for position in group:
            questions[position].question_group_id = group_id

    return len(groups)
