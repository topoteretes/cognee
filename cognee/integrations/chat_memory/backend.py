"""The memory backend: the seam between the adapter and cognee.

The adapter never calls cognee directly. It talks to a :class:`MemoryBackend`,
a four-method interface expressed entirely in the plain value objects from
:mod:`.models`. This gives two things:

* **Deterministic tests with no keys.** The suite runs the real adapter against
  an in-memory fake backend, so ``ingest`` / ``answer`` / ``forget`` round-trips
  are exercised without an LLM, embeddings, or a database.
* **Pluggable transport.** :class:`CogneeMemoryBackend` is the in-process
  Python-SDK reference implementation (the fast path, no network hop). A
  TS-client or MCP-backed backend can satisfy the same contract later without
  touching the adapter or any bot.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid5, NAMESPACE_URL

from cognee.shared.logging_utils import get_logger

from .models import RecalledItem

logger = get_logger("chat_memory.backend")


@runtime_checkable
class MemoryBackend(Protocol):
    """What the adapter needs from a memory system. Four methods, no cognee types."""

    async def remember(
        self,
        text: str,
        *,
        dataset: str,
        session: str,
        external_metadata: dict[str, Any],
        item_id: Optional[str] = None,
    ) -> None:
        """Store ``text`` in ``dataset`` (+ fast ``session`` cache), returning fast.

        ``external_metadata`` is stamped onto the stored item; it carries the
        author and permalink that later power both per-user forget and
        citations. ``item_id`` is a stable, caller-chosen id for idempotency.
        """
        ...

    async def recall(
        self, query: str, *, dataset: str, session: str, top_k: int
    ) -> list[RecalledItem]:
        """Recall from the ``session`` cache first, then the ``dataset`` graph."""
        ...

    async def forget_scope(self, *, dataset: str) -> dict:
        """Wipe an entire ``dataset`` (the whole-scope 'forget everything here')."""
        ...

    async def forget_user(self, *, dataset: str, user: str) -> dict:
        """Forget one user's items inside a shared ``dataset`` (their 'forget me')."""
        ...


def deterministic_item_id(*parts: str) -> str:
    """A stable UUIDv5 string from key parts (e.g. channel, user, timestamp).

    Re-ingesting the same message yields the same id, so the backend can dedup
    instead of storing duplicates (captures often replay on reconnect).
    """
    return str(uuid5(NAMESPACE_URL, "|".join(parts)))


# A tiny stopword set so the lexical ranker keys off content words, not
# grammar. Keeps the in-memory demo/tests from matching on "the"/"on"/etc.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to "
    "was were will with we you your my our i do does when what who where how".split()
)


def _content_tokens(text: str) -> set[str]:
    import re

    return {w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) >= 2 and w not in _STOPWORDS}


def _keyword_overlap(query: str, text: str) -> int:
    """Count shared content-word tokens. A tiny deterministic ranker."""
    return len(_content_tokens(query) & _content_tokens(text))


class InMemoryMemoryBackend:
    """A dependency-free :class:`MemoryBackend` for local dev, demos, and tests.

    Keeps everything in process dictionaries and ranks recall by keyword
    overlap, so a bot (and this package's test suite) can exercise the full
    ``ingest`` -> ``answer`` -> ``forget`` contract with no LLM, embeddings,
    database, or API keys. It faithfully models the two behaviours the adapter
    depends on: dedup by ``item_id`` and per-user forget by the ``user`` field
    stamped into ``external_metadata``.

    It is intentionally not a knowledge graph: recall is lexical, so swap in
    :class:`CogneeMemoryBackend` for real multi-hop memory.
    """

    def __init__(self) -> None:
        # dataset -> item_id -> stored record
        self._store: dict[str, dict[str, dict[str, Any]]] = {}

    async def remember(
        self,
        text: str,
        *,
        dataset: str,
        session: str,
        external_metadata: dict[str, Any],
        item_id: Optional[str] = None,
    ) -> None:
        key = item_id or deterministic_item_id(dataset, session, text)
        # Dedup: a replayed message with the same id overwrites, never duplicates.
        self._store.setdefault(dataset, {})[key] = {
            "text": text,
            "session": session,
            "external_metadata": dict(external_metadata),
        }

    async def recall(
        self, query: str, *, dataset: str, session: str, top_k: int
    ) -> list[RecalledItem]:
        records = self._store.get(dataset, {})
        scored: list[tuple[int, RecalledItem]] = []
        for record in records.values():
            hits = _keyword_overlap(query, record["text"])
            if hits == 0:
                continue
            stamp = record["external_metadata"]
            scored.append(
                (
                    hits,
                    RecalledItem(
                        text=record["text"],
                        # Same-session items came from the fast cache; others
                        # are recalled from the shared dataset graph.
                        source="session" if record["session"] == session else "graph",
                        score=float(hits),
                        permalink=stamp.get("permalink"),
                        user=str(stamp["user"]) if stamp.get("user") is not None else None,
                        metadata=stamp,
                    ),
                )
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    async def forget_scope(self, *, dataset: str) -> dict:
        removed = len(self._store.pop(dataset, {}))
        return {"dataset": dataset, "items_removed": removed, "status": "success"}

    async def forget_user(self, *, dataset: str, user: str) -> dict:
        records = self._store.get(dataset, {})
        to_remove = [
            key
            for key, record in records.items()
            if str(record["external_metadata"].get("user")) == str(user)
        ]
        for key in to_remove:
            del records[key]
        return {
            "dataset": dataset,
            "user": user,
            "items_removed": len(to_remove),
            "status": "success",
        }


class CogneeMemoryBackend:
    """In-process reference backend backed by the cognee Python SDK.

    Maps the adapter's four methods onto ``cognee.remember`` / ``cognee.recall``
    / ``cognee.forget`` and nothing else. All cognee imports are local to the
    methods so importing the adapter package never triggers cognee's heavier
    import graph.

    Args:
        run_in_background: Pass-through to ``remember``. ``True`` (default) makes
            ingestion fire-and-forget: the call returns immediately and the
            session-to-graph bridge proceeds in the background.
        top_k: Default recall breadth when the adapter does not override it.
    """

    def __init__(self, *, run_in_background: bool = True, top_k: int = 10) -> None:
        self.run_in_background = run_in_background
        self.top_k = top_k

    async def remember(
        self,
        text: str,
        *,
        dataset: str,
        session: str,
        external_metadata: dict[str, Any],
        item_id: Optional[str] = None,
    ) -> None:
        import cognee
        from cognee.tasks.ingestion.data_item import DataItem

        data_id: Optional[UUID] = None
        if item_id:
            try:
                data_id = UUID(item_id)
            except (ValueError, AttributeError, TypeError):
                # Not already a UUID, so derive a stable one so ingestion is
                # idempotent on the caller's opaque id.
                data_id = uuid5(NAMESPACE_URL, item_id)

        item = DataItem(data=text, external_metadata=external_metadata, data_id=data_id)
        await cognee.remember(
            item,
            dataset_name=dataset,
            session_id=session,
            run_in_background=self.run_in_background,
        )

    async def recall(
        self, query: str, *, dataset: str, session: str, top_k: int
    ) -> list[RecalledItem]:
        import cognee

        responses = await cognee.recall(
            query,
            datasets=[dataset],
            session_id=session,
            top_k=top_k,
            include_references=True,
        )
        return [self._to_recalled_item(response) for response in responses]

    async def forget_scope(self, *, dataset: str) -> dict:
        import cognee

        return await cognee.forget(dataset=dataset)

    async def forget_user(self, *, dataset: str, user: str) -> dict:
        """Delete only ``user``'s items from a shared ``dataset``.

        Resolves the user's ``Data`` rows by the ``user`` field stamped into
        ``external_metadata`` at ingest, then forgets them one by one.

        Scope note (agreed on the tracking issue): this removes the user's own
        items. When cognify has merged facts from several users into one shared
        graph node, dropping those items can leave the shared node partially
        referenced. Fully dedup-aware deletion, which removes a node or edge
        only when no other user's data still references it (the
        ``get_unique_nodes_for_data`` / ``get_unique_edges_for_data`` approach
        from cognee-rs #36), belongs in the core and is the planned follow-up.
        Per-user datasets (e.g. the second brain's ``brain:{user}``) are not
        affected: their "forget me" is a whole-dataset wipe with nothing shared
        to orphan.
        """
        import cognee

        dataset_id, data_ids = await self._resolve_user_data_ids(dataset, user)
        if not data_ids:
            return {"dataset": dataset, "user": user, "items_removed": 0, "status": "success"}

        removed = 0
        for data_id in data_ids:
            try:
                await cognee.forget(data_id=data_id, dataset_id=dataset_id, memory_only=True)
                removed += 1
            except Exception as exc:  # pragma: no cover - defensive per-item guard
                logger.warning(
                    "chat_memory: forget_user failed for data_id=%s in dataset=%s: %s",
                    data_id,
                    dataset,
                    exc,
                )
        return {
            "dataset": dataset,
            "user": user,
            "items_removed": removed,
            "status": "success",
        }

    async def _resolve_user_data_ids(
        self, dataset: str, user: str
    ) -> tuple[Optional[UUID], list[UUID]]:
        """Return ``(dataset_id, [data_id, ...])`` for items authored by ``user``."""
        import cognee
        from cognee.modules.data.methods.get_authorized_dataset_by_name import (
            get_authorized_dataset_by_name,
        )
        from cognee.modules.users.methods import get_default_user

        default_user = await get_default_user()
        try:
            resolved = await get_authorized_dataset_by_name(dataset, default_user, "delete")
        except Exception:
            resolved = None
        if resolved is None:
            return None, []

        rows = await cognee.datasets.list_data(resolved.id, user=default_user)
        matches: list[UUID] = []
        for row in rows or []:
            metadata = getattr(row, "external_metadata", None)
            if isinstance(metadata, dict) and str(metadata.get("user")) == str(user):
                matches.append(row.id)
        return resolved.id, matches

    @staticmethod
    def _to_recalled_item(response: Any) -> RecalledItem:
        """Normalize a cognee ``RecallResponse`` (a tagged union) to a RecalledItem.

        Handled generically via attribute access so it survives the union
        growing new member types: every member carries ``source``; graph
        entries carry ``text``/``score``/``metadata``; session entries carry
        ``answer``/``content``. The ``external_metadata`` stamp, surfaced under
        an item's ``metadata``, yields the permalink and author for citations.
        """
        source = getattr(response, "source", "graph")

        text = (
            getattr(response, "text", None)
            or getattr(response, "answer", None)
            or getattr(response, "content", None)
            or ""
        )

        metadata = getattr(response, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        # cognee stamps external_metadata under raw / metadata depending on the
        # retriever; check both so citations resolve regardless of source.
        stamp = metadata.get("external_metadata")
        if not isinstance(stamp, dict):
            raw = getattr(response, "raw", None)
            if isinstance(raw, dict):
                candidate = raw.get("external_metadata")
                stamp = candidate if isinstance(candidate, dict) else {}
            else:
                stamp = {}

        author = stamp.get("user")
        return RecalledItem(
            text=text,
            source=source,
            score=getattr(response, "score", None),
            permalink=stamp.get("permalink"),
            user=str(author) if author is not None else None,
            metadata=metadata or stamp,
        )
