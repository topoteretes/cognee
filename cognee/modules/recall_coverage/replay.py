"""Replay each question row against memory, as that row's own user.

Phase 3 step 11 of the recall-coverage spec. One search per question row, and
four rules that are each the difference between a real number and a fabricated
one:

* **As the row's own user, never as the caller.** A run is tenant-wide, so most
  rows belong to somebody else. Replaying Ben's question as Anna answers it out
  of Anna's brains and then labels the result with Ben's ``user_id`` — a report
  that is wrong in a way nothing downstream can detect. Each row's ``User`` is
  loaded (once per run, via :class:`ReplayUserCache`) and the search runs as
  them. ``authorized_search`` then resolves *their* readable datasets and
  ``search_in_datasets_context`` sets the per-dataset database context itself, so
  this module must not touch ``set_database_global_context_variables``.
* **``authorized_search`` only — never ``search()`` and never ``recall()``.**
  Both of those log: ``search()`` calls ``log_search_history`` after the search
  (``cognee/modules/search/methods/search.py``) and so does ``recall()``
  (``cognee/api/v1/recall/recall.py``). A replay routed through either writes new
  ``queries`` rows, which land inside the *next* run's window and get replayed
  again — a feedback loop that inflates every count for free.
  ``COGNEE_LOG_SEARCH_HISTORY`` is not a defence: ``_LOG_ENABLED`` in
  ``log_query.py``/``log_result.py`` is a module-level constant evaluated at
  import, so it cannot be flipped inside a running API process.
* **``session_user`` is reset to ``None`` for the replay.**
  ``GraphCompletionRetriever._use_session_cache()`` is
  ``bool(user_id and CacheConfig().caching)``, and ``recall()`` sets that
  ContextVar (``set_session_user_context_variable``). Left alone, a run triggered
  from inside an authenticated request with ``CACHING=true`` would write a
  session-cache QA turn per replayed question and could fire an AUTO_FEEDBACK
  LLM call each time — the run would be editing the memory it is measuring.
* **Retrieval only: ``only_context=True``.** The completion is a judge-phase
  call, made only for rows that scored above zero (spec section 2 phase 3 step
  12). Letting the retriever generate one here would bill a completion for every
  row including the ones about to score 0, which is the cost the judge's ordering
  exists to avoid. It also skips ``prepare_session_turn_for_retrieval``
  (``get_retriever_output.py`` line 57), a second session-cache write path.

A per-row failure records ``error`` and leaves the scores NULL rather than
failing the run: one unreadable dataset must not throw away a hundred judged
rows.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator, Optional, Sequence
from uuid import UUID

from cognee.context_global_variables import session_user
from cognee.modules.recall_coverage.dedup import DedupedQuestion
from cognee.modules.recall_coverage.types import CoverageParams
from cognee.modules.search.methods.search import authorized_search
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_user
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")

# The search type every row is replayed under. Not a config parameter: it is the
# retriever whose answers the judge's rubric is written about, and a run whose
# rows were retrieved by different strategies would report a mean over
# incomparable things. The *observed* query types that enter the window are a
# separate, configured set (``RecallCoverageConfig.query_types``).
REPLAY_QUERY_TYPE = SearchType.GRAPH_COMPLETION

# Joins several payloads' contexts, and separates them inside one payload's
# list-shaped context.
CONTEXT_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ReplayedRow:
    """What memory retrieved for one question row.

    ``retrieval_context`` is ``None`` both when the search returned nothing
    usable and when it failed, so the column is NULL rather than an empty string
    in either case; ``error`` is what distinguishes them. A row with an ``error``
    is never judged — its scores stay NULL, because "we could not ask" is not
    evidence that memory could not answer.
    """

    retrieval_context: Optional[str]
    dataset_name: Optional[str]
    payload_count: int
    error: Optional[str] = None

    @property
    def has_context(self) -> bool:
        return bool(self.retrieval_context)


@contextmanager
def without_session_user() -> Iterator[None]:
    """Run a block with the ``session_user`` ContextVar cleared.

    ``set_session_user_context_variable`` returns no token, so the token is taken
    here and reset in ``finally``: a replay must not leave the surrounding
    request without its user, and it must not run with one. See the module
    docstring for what the retrievers do with it.
    """
    token = session_user.set(None)
    try:
        yield
    finally:
        session_user.reset(token)


def flatten_context(context: Any) -> str:
    """Render a payload's ``context`` as the text a judge and a human can read.

    ``SearchResultPayload.context`` is ``str | list[str] | None``. A plain
    ``str(context)`` of the list shape would store — and prompt with — a Python
    repr (``"['a', 'b']"``), so the list is joined instead. For the ``str`` shape
    this is exactly ``str(context)`` stripped.
    """
    if context is None:
        return ""
    if isinstance(context, str):
        return context.strip()
    if isinstance(context, (list, tuple)):
        parts = [flatten_context(item) for item in context]
        return CONTEXT_SEPARATOR.join(part for part in parts if part)
    return str(context).strip()


class ReplayUserCache:
    """Load each distinct ``user_id``'s ``User`` once per run.

    Many rows share a user — a tenant-wide window over 150 questions typically
    has a handful of askers — and every load is a relational round trip with two
    eager-loaded relationships. Failures are cached too: a ``user_id`` whose row
    is gone (``get_user`` raises ``EntityNotFoundError``, it does not return
    ``None``) must not be retried once per question.

    ``loader`` is injectable for tests. The lock serialises first-time loads,
    which is deliberate: they are fast, the replays they gate are not, and it
    keeps two concurrent rows from issuing the same query.
    """

    def __init__(self, loader: Optional[Callable[[UUID], Awaitable[Any]]] = None) -> None:
        self._loader = loader or get_user
        self._lock = asyncio.Lock()
        self._users: dict[UUID, Any] = {}
        self._failures: dict[UUID, Exception] = {}

    @property
    def loaded_count(self) -> int:
        """Distinct users actually loaded. Asserted on by the cache's test."""
        return len(self._users)

    async def get(self, user_id: UUID) -> Any:
        async with self._lock:
            if user_id in self._users:
                return self._users[user_id]
            if user_id in self._failures:
                raise self._failures[user_id]

            try:
                user = await self._loader(user_id)
            except Exception as error:
                self._failures[user_id] = error
                raise

            self._users[user_id] = user
            return user


async def replay_question(
    question: DedupedQuestion,
    user: Any,
    *,
    replay_top_k: int,
    store_context_max_chars: int,
    search: Optional[Callable[..., Awaitable[Any]]] = None,
) -> ReplayedRow:
    """Retrieve context for one question row, as ``user``.

    ``dataset_ids`` branches on the **row's** ``dataset_id``, not on its
    ``source``: a row scoped to one dataset is replayed against that dataset, and
    a row with no dataset attribution — a search that spanned several, or a
    curated question that merged with nothing — is replayed against everything
    the user can read, which is what the original ask did. A curated question
    that merged into dataset D therefore replays against D, the same memory the
    observed rows in that partition were answered from.

    Zero payloads is a reachable, non-exceptional outcome:
    ``get_authorized_existing_datasets`` returns ``[]`` when the row's dataset is
    not readable by the row's user, and the search then returns no payloads at
    all. That is an empty context, which the judge defines as score 0.
    """
    search_fn = search or authorized_search
    dataset_ids = [question.dataset_id] if question.dataset_id else None

    with without_session_user():
        payloads = await search_fn(
            query_type=REPLAY_QUERY_TYPE,
            query_text=question.text,
            user=user,
            dataset_ids=dataset_ids,
            top_k=replay_top_k,
            session_id=None,
            only_context=True,
        )

    payloads = list(payloads or [])
    parts = [flatten_context(getattr(payload, "context", None)) for payload in payloads]
    joined = CONTEXT_SEPARATOR.join(part for part in parts if part)

    return ReplayedRow(
        retrieval_context=joined[:store_context_max_chars] or None,
        # Only meaningful when the row resolved to exactly one dataset; a row
        # replayed across everything readable has no single dataset to name, and
        # its ``dataset_id`` is NULL for the same reason.
        dataset_name=getattr(payloads[0], "dataset_name", None) if len(payloads) == 1 else None,
        payload_count=len(payloads),
    )


async def replay_questions(
    questions: Sequence[DedupedQuestion],
    *,
    params: CoverageParams,
    user_cache: Optional[ReplayUserCache] = None,
    search: Optional[Callable[..., Awaitable[Any]]] = None,
) -> list[ReplayedRow]:
    """Replay every question row, index-aligned with ``questions``.

    Bounded by ``asyncio.Semaphore(replay_max_concurrent)``: each row is a graph
    retrieval against a possibly-embedded per-dataset database, so an unbounded
    fan-out over 150 rows would open 150 dataset contexts at once.

    Every exception is caught per row and recorded as that row's ``error`` — a
    missing user, an unreadable dataset, a retriever failure. The run continues
    and those rows report NULL scores.
    """
    if not questions:
        return []

    cache = user_cache if user_cache is not None else ReplayUserCache()
    semaphore = asyncio.Semaphore(max(1, params.replay_max_concurrent))

    async def _replay_one(question: DedupedQuestion) -> ReplayedRow:
        async with semaphore:
            try:
                user = await cache.get(question.user_id)
                return await replay_question(
                    question,
                    user,
                    replay_top_k=params.replay_top_k,
                    store_context_max_chars=params.store_context_max_chars,
                    search=search,
                )
            except Exception as error:
                logger.warning(
                    "recall_coverage: replay failed for a question of user %s on dataset %s: %s",
                    question.user_id,
                    question.dataset_id,
                    error,
                )
                return ReplayedRow(
                    retrieval_context=None,
                    dataset_name=None,
                    payload_count=0,
                    error=str(error) or type(error).__name__,
                )

    return list(await asyncio.gather(*(_replay_one(question) for question in questions)))


__all__ = [
    "CONTEXT_SEPARATOR",
    "REPLAY_QUERY_TYPE",
    "ReplayUserCache",
    "ReplayedRow",
    "flatten_context",
    "replay_question",
    "replay_questions",
    "without_session_user",
]
