"""Thin adapter wrapping cognee's remember/recall/forget APIs.

Until issue #3608 (shared ChatMemoryAdapter) lands, this module provides
a small adapter that maps support-thread domain concepts to the verified
cognee API parameters.

It also manages the **thread_id ↔ cognee data_id** mapping required to
route ``!forget`` commands to the correct cognee UUID.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from config import BotConfig
from models import Citation, SupportThread

logger = logging.getLogger(__name__)


class MemoryAdapter:
    """Adapter between support-triage domain and cognee memory APIs."""

    def __init__(self, config: BotConfig) -> None:
        self._config = config
        # In-memory mapping: platform thread_id → cognee data_id (UUID).
        # Production deployments should persist this to a database.
        self._thread_id_to_data_id: dict[str, UUID] = {}

    # ── Ingest ──────────────────────────────────────────────────────────

    async def ingest_resolved_thread(self, thread: SupportThread) -> dict:
        """Ingest a resolved support thread into cognee memory.

        Calls ``cognee.remember()`` with the thread rendered as a
        structured document. Extracts the cognee-generated ``data_id``
        from the ``RememberResult`` and persists the mapping.

        Returns:
            Dict with ``status``, ``data_id``, and ``dataset_name``.
        """
        import cognee

        doc_text = thread.to_document()
        session_id = self._config.session_id_for(thread.channel_id)

        result = await cognee.remember(
            doc_text,
            self._config.dataset_name,
            session_id=session_id,
        )

        # Extract cognee-generated data_id from RememberResult.items
        data_id: Optional[UUID] = None
        if result.items:
            raw_id = result.items[0].get("id")
            if raw_id is not None:
                data_id = UUID(str(raw_id))
                self._thread_id_to_data_id[thread.thread_id] = data_id
                logger.info(
                    "Persisted mapping: thread_id=%s → data_id=%s",
                    thread.thread_id,
                    data_id,
                )

        return {
            "status": result.status,
            "data_id": str(data_id) if data_id else None,
            "dataset_name": result.dataset_name,
            "items_processed": result.items_processed,
        }

    # ── Recall ──────────────────────────────────────────────────────────

    async def find_similar_issues(
        self, query: str, channel_id: str
    ) -> list[Citation]:
        """Search cognee memory for similar past support issues.

        Calls ``cognee.recall()`` with the query text and returns
        ranked ``Citation`` objects built from the graph results.

        Args:
            query: The natural-language support question.
            channel_id: Channel to scope the search (when channel-scoped).

        Returns:
            List of ``Citation`` objects, ranked by relevance.
        """
        import cognee

        session_id = self._config.session_id_for(channel_id)

        results = await cognee.recall(
            query,
            datasets=[self._config.dataset_name],
            session_id=session_id,
            top_k=self._config.top_k,
        )

        citations: list[Citation] = []
        for entry in results:
            # ResponseGraphEntry carries text, score, metadata
            text = getattr(entry, "text", "")
            score = getattr(entry, "score", None)
            metadata = getattr(entry, "metadata", {}) or {}

            # Apply minimum relevance threshold
            if (
                score is not None
                and self._config.min_relevance_score > 0
                and score < self._config.min_relevance_score
            ):
                continue

            citation = Citation(
                source_thread_id=metadata.get("thread_id", "unknown"),
                thread_url=metadata.get("thread_url", ""),
                resolution_summary=text,
                similarity_score=score,
            )
            citations.append(citation)

        return citations

    # ── Forget ──────────────────────────────────────────────────────────

    async def forget_thread(self, thread_id: str) -> dict:
        """Remove a specific thread from cognee memory.

        Looks up the cognee ``data_id`` UUID from the internal mapping,
        then calls ``cognee.forget(data_id=..., dataset=...)``.

        Args:
            thread_id: The platform-specific thread identifier.

        Returns:
            Dict with deletion result or error info.

        Raises:
            KeyError: If the thread was never ingested (no mapping exists).
        """
        import cognee

        if thread_id not in self._thread_id_to_data_id:
            raise KeyError(
                f"Thread '{thread_id}' was never ingested — "
                f"no data_id mapping exists. Cannot forget."
            )

        data_id = self._thread_id_to_data_id[thread_id]
        result = await cognee.forget(
            data_id=data_id,
            dataset=self._config.dataset_name,
        )

        # Remove the mapping after successful deletion
        del self._thread_id_to_data_id[thread_id]
        logger.info("Forgot thread_id=%s (data_id=%s)", thread_id, data_id)

        return result

    async def forget_all(self) -> dict:
        """Remove all support-thread data from cognee memory.

        Calls ``cognee.forget(everything=True)`` and clears all mappings.
        """
        import cognee

        result = await cognee.forget(everything=True)
        self._thread_id_to_data_id.clear()
        logger.info("Forgot everything — all mappings cleared")
        return result

    # ── Mapping accessors (for testing / debugging) ─────────────────────

    def get_data_id(self, thread_id: str) -> Optional[UUID]:
        """Return the cognee data_id for a thread, or None."""
        return self._thread_id_to_data_id.get(thread_id)

    def has_mapping(self, thread_id: str) -> bool:
        """Check if a thread_id has been mapped to a cognee data_id."""
        return thread_id in self._thread_id_to_data_id
