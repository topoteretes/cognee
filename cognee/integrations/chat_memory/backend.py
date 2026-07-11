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

from .models import Citation

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
        """Store ``text`` durably in ``dataset``, returning fast.

        ``external_metadata`` is stamped onto the stored item; it carries the
        author and permalink that later power both per-user forget and
        citations. ``item_id`` is a stable, caller-chosen id for idempotency.
        ``session`` identifies the live conversation; a backend may use it for a
        recency cache (the in-memory one does) or ignore it.
        """
        ...

    async def recall(self, query: str, *, dataset: str, session: str, top_k: int) -> list[Citation]:
        """Recall the most relevant items for ``query`` from ``dataset``."""
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
        key = item_id or deterministic_item_id(dataset, text)
        # Dedup: a replayed message with the same id overwrites, never duplicates.
        self._store.setdefault(dataset, {})[key] = {
            "text": text,
            "session": session,
            "external_metadata": dict(external_metadata),
        }

    async def recall(self, query: str, *, dataset: str, session: str, top_k: int) -> list[Citation]:
        records = self._store.get(dataset, {})
        scored: list[tuple[int, Citation]] = []
        for record in records.values():
            hits = _keyword_overlap(query, record["text"])
            if hits == 0:
                continue
            stamp = record["external_metadata"]
            scored.append(
                (
                    hits,
                    Citation(
                        text=record["text"],
                        # Same-session items came from the fast cache; others
                        # are recalled from the shared dataset graph.
                        source="session" if record["session"] == session else "graph",
                        score=float(hits),
                        permalink=stamp.get("permalink"),
                        user=str(stamp["user"]) if stamp.get("user") is not None else None,
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
    / ``cognee.forget`` and nothing else. Those entrypoints are imported lazily
    inside each method, so the SDK's operational import graph is only pulled in
    when a primitive actually runs.

    Storage is durable: a message is ingested through cognee's permanent
    ``add()`` + ``cognify()`` path (``run_in_background=True`` keeps it
    fire-and-forget). That path is the only one that writes ``external_metadata``
    onto the ``Data`` row and honours a caller-set ``data_id`` — which is exactly
    what per-user "forget me" and citation resolution rely on. cognee's
    ``session_id`` path is a session-cache-only fast path that drops both, so
    this backend does not use it; a session recency cache is a future add-on.

    Args:
        run_in_background: Pass-through to ``remember``. ``True`` (default) makes
            ingestion fire-and-forget: the call returns immediately and the
            add/cognify build proceeds in the background.
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

        item = DataItem(
            data=text,
            external_metadata=external_metadata,
            data_id=self._as_uuid(item_id),
        )
        # Durable path (no session_id): add() writes external_metadata + honours
        # data_id onto the Data row and cognify builds the graph.
        await cognee.remember(
            item,
            dataset_name=dataset,
            run_in_background=self.run_in_background,
        )

    async def recall(self, query: str, *, dataset: str, session: str, top_k: int) -> list[Citation]:
        import cognee

        responses = await cognee.recall(
            query,
            datasets=[dataset],
            top_k=top_k,
            include_references=True,
        )
        if not responses:
            return []
        # recall reports each hit's origin as a data_id (the ingested Data.id);
        # the permalink/author live in the stamp we wrote onto that Data row.
        stamps = await self._stamps_by_data_id(dataset)
        return [self._to_citation(response, stamps) for response in responses]

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

        dataset_id, rows = await self._dataset_rows(dataset)
        data_ids = [
            row.id
            for row in rows
            if isinstance(getattr(row, "external_metadata", None), dict)
            and str(row.external_metadata.get("user")) == str(user)
        ]
        if not data_ids:
            return {"dataset": dataset, "user": user, "items_removed": 0, "status": "success"}

        removed = 0
        for data_id in data_ids:
            try:
                # Full delete (not memory_only): "forget me" must drop the raw
                # Data record too, not just its graph/vector projection.
                await cognee.forget(data_id=data_id, dataset_id=dataset_id)
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

    async def _dataset_rows(self, dataset: str) -> tuple[Optional[UUID], list[Any]]:
        """Resolve ``(dataset_id, Data rows)`` for ``dataset``; ``(None, [])`` if absent.

        Shared by forget-me and citation resolution: both need the ``Data`` rows
        that carry the ``external_metadata`` stamp written at ingest.
        """
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
        return resolved.id, list(rows or [])

    async def _stamps_by_data_id(self, dataset: str) -> dict[str, dict[str, Any]]:
        """Map ``str(Data.id) -> external_metadata`` for citation resolution.

        Returns ``{}`` when the dataset can't be resolved, so citations degrade
        to text-only rather than failing.
        """
        _, rows = await self._dataset_rows(dataset)
        stamps: dict[str, dict[str, Any]] = {}
        for row in rows:
            stamp = getattr(row, "external_metadata", None)
            if isinstance(stamp, dict):
                stamps[str(row.id)] = stamp
        return stamps

    @staticmethod
    def _as_uuid(item_id: Optional[str]) -> Optional[UUID]:
        """Coerce a caller item id to the UUID cognee uses as ``Data.id``.

        Accepts a real UUID string as-is; derives a stable UUIDv5 from any other
        opaque id so ingestion stays idempotent on the caller's key.
        """
        if not item_id:
            return None
        try:
            return UUID(item_id)
        except (ValueError, AttributeError, TypeError):
            return uuid5(NAMESPACE_URL, item_id)

    @staticmethod
    def _to_citation(response: Any, stamps: dict[str, dict[str, Any]]) -> Citation:
        """Normalize a cognee ``RecallResponse`` (a tagged union) to a Citation.

        Handled generically via attribute access so it survives the union
        growing new member types: every member carries ``source``; graph
        entries carry ``text``/``score``/``metadata``; session entries carry
        ``answer``/``content``. When a result carries a ``data_id`` in its
        provenance metadata, the matching stamp yields the permalink and author
        for a citation; otherwise the citation is text-only.
        """
        source = getattr(response, "source", "graph")
        text = (
            getattr(response, "text", None)
            or getattr(response, "answer", None)
            or getattr(response, "content", None)
            or ""
        )

        metadata = getattr(response, "metadata", None)
        data_id = metadata.get("data_id") if isinstance(metadata, dict) else None
        stamp = stamps.get(str(data_id), {}) if data_id is not None else {}
        author = stamp.get("user")
        return Citation(
            text=text,
            source=source,
            score=getattr(response, "score", None),
            permalink=stamp.get("permalink"),
            user=str(author) if author is not None else None,
        )
