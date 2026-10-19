"""Filesystem-backed cache adapter that mirrors session QA turns into a Headroom
memory store (https://github.com/headroomlabs-ai/headroom).

The FS adapter remains the source of truth for session state. Every QA entry is
additionally saved to Headroom's hierarchical memory (``headroom.memory``), scoped
by cognee's ``user_id`` and ``session_id``, so cognee sessions become part of the
one memory store Headroom shares across the agents it wraps (Claude Code, Codex,
Gemini, Grok, ...). Headroom's local backend is embedded (SQLite + sqlite-vec + a
local ONNX embedder by default) and makes no LLM calls; the mirror never leaves
the machine unless the ``openai`` embedder is configured.

Unlike tapes, Headroom memories are addressable, so the mirror is kept in sync:
a deleted QA entry or session removes the matching memories, and an update that
changes the question or answer replaces its memory. Feedback-only updates,
agent-trace steps, session-context entries, and usage logs stay local. ``prune()``
removes only the memories this adapter wrote -- the Headroom store is shared with
other agents and is never wiped wholesale.

The ``qa_id -> memory_id`` map that makes this possible lives in the FS cache
without an expiry: Headroom memories never expire, so the map must outlive the
session TTL or an expired session's turns could no longer be removed through
cognee. A map entry is dropped only once its memory is deleted; a delete that
fails keeps its entry so a later delete or ``prune()`` retries it.

Mirror failures are logged and swallowed so the FS write remains authoritative.
"""

import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any

from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter
from cognee.infrastructure.databases.exceptions.exceptions import HeadroomNotInstalledError
from cognee.shared.logging_utils import get_logger

logger = get_logger("HeadroomCacheAdapter")

MEMORY_SOURCE = "cognee"


class HeadroomCacheAdapter(FSCacheAdapter):
    """FS adapter that also mirrors each QA turn into Headroom's memory store."""

    def __init__(
        self,
        session_ttl_seconds: int | None = 604800,
        *,
        headroom_db_path: str | None = None,
        headroom_embedder: str = "onnx",
        headroom_embedder_model: str | None = None,
        headroom_embedder_api_key: str | None = None,
        headroom_ollama_base_url: str = "http://localhost:11434",
        headroom_vector_dimension: int = 384,
        headroom_agent_name: str = "cognee",
    ):
        _ensure_headroom_installed()
        super().__init__(session_ttl_seconds=session_ttl_seconds)
        self.headroom_db_path = headroom_db_path
        self.headroom_embedder = headroom_embedder
        self.headroom_embedder_model = headroom_embedder_model
        self.headroom_embedder_api_key = headroom_embedder_api_key
        self.headroom_ollama_base_url = headroom_ollama_base_url
        self.headroom_vector_dimension = headroom_vector_dimension
        self.headroom_agent_name = headroom_agent_name
        self._headroom_backend: Any = None

    # ------------------------------------------------------------------
    # Headroom backend
    # ------------------------------------------------------------------

    def _resolve_db_path(self) -> Path:
        """Explicit path wins; otherwise Headroom's own shared workspace store."""
        if self.headroom_db_path:
            db_path = Path(self.headroom_db_path)
        else:
            from headroom import paths as headroom_paths

            db_path = headroom_paths.memory_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def _get_backend(self):
        """Construct Headroom's LocalBackend once; it initializes itself on first use."""
        if self._headroom_backend is None:
            from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

            config_kwargs: dict[str, Any] = {
                "db_path": str(self._resolve_db_path()),
                "embedder_backend": self.headroom_embedder,
                "vector_dimension": self.headroom_vector_dimension,
                "openai_api_key": self.headroom_embedder_api_key,
                "ollama_base_url": self.headroom_ollama_base_url,
            }
            if self.headroom_embedder_model:
                config_kwargs["embedder_model"] = self.headroom_embedder_model

            self._headroom_backend = LocalBackend(LocalBackendConfig(**config_kwargs))
            logger.debug(
                "HeadroomCacheAdapter mirroring to %s (embedder=%s)",
                config_kwargs["db_path"],
                self.headroom_embedder,
            )
        return self._headroom_backend

    # ------------------------------------------------------------------
    # qa_id -> headroom memory id bookkeeping (kept in the FS cache)
    # ------------------------------------------------------------------

    _MEMORY_IDS_PREFIX = "headroom_memory_ids:"

    @classmethod
    def _memory_ids_key(cls, user_id: str, session_id: str) -> str:
        return f"{cls._MEMORY_IDS_PREFIX}{user_id}:{session_id}"

    def _load_memory_ids(self, user_id: str, session_id: str) -> dict[str, str]:
        value = self.cache.get(self._memory_ids_key(user_id, session_id))
        return json.loads(value) if value else {}

    def _write_memory_ids(self, key: str, mapping: dict[str, str]) -> None:
        # No expiry: the memories this map points at never expire in Headroom,
        # so the map must survive the session TTL for them to stay deletable.
        if not mapping:
            self.cache.delete(key)
            return
        self.cache.set(key, json.dumps(mapping))

    def _save_memory_ids(self, user_id: str, session_id: str, mapping: dict[str, str]) -> None:
        self._write_memory_ids(self._memory_ids_key(user_id, session_id), mapping)

    def _record_memory_id(self, user_id: str, session_id: str, qa_id: str, memory_id: str) -> None:
        with self.cache.transact():
            mapping = self._load_memory_ids(user_id, session_id)
            mapping[qa_id] = memory_id
            self._save_memory_ids(user_id, session_id, mapping)

    def _restore_memory_ids(self, key: str, failed: dict[str, str]) -> None:
        """Put back the entries of memories that could not be deleted, so they are retried."""
        if not failed:
            return
        with self.cache.transact():
            value = self.cache.get(key)
            mapping = json.loads(value) if value else {}
            mapping.update(failed)
            self._write_memory_ids(key, mapping)

    @staticmethod
    def _stale_key(qa_id: str, memory_id: str) -> str:
        """Map key for a replaced memory whose delete failed; still owned by ``qa_id``."""
        return f"{qa_id}@{memory_id}"

    def _pop_memory_ids(self, user_id: str, session_id: str, qa_id: str) -> dict[str, str]:
        """Pop the entry's current memory plus any replaced ones still awaiting deletion."""
        stale_prefix = self._stale_key(qa_id, "")
        with self.cache.transact():
            mapping = self._load_memory_ids(user_id, session_id)
            popped = {
                key: mapping.pop(key)
                for key in list(mapping)
                if key == qa_id or key.startswith(stale_prefix)
            }
            self._save_memory_ids(user_id, session_id, mapping)
        return popped

    def _pop_all_memory_ids(self, user_id: str, session_id: str) -> dict[str, str]:
        with self.cache.transact():
            mapping = self._load_memory_ids(user_id, session_id)
            self.cache.delete(self._memory_ids_key(user_id, session_id))
        return mapping

    # ------------------------------------------------------------------
    # Mirroring
    # ------------------------------------------------------------------

    @staticmethod
    def _build_memory_content(question: str, answer: str) -> str:
        return f"Q: {question}\nA: {answer}"

    def _build_memory_metadata(self, user_id: str, session_id: str, qa_id: str) -> dict:
        return {
            "source": MEMORY_SOURCE,
            "agent": self.headroom_agent_name,
            "cognee_user_id": user_id,
            "cognee_session_id": session_id,
            "qa_id": qa_id,
        }

    async def _mirror_qa_to_headroom(
        self, *, user_id: str, session_id: str, qa_id: str, question: str, answer: str
    ) -> None:
        try:
            backend = self._get_backend()
            memory = await backend.save_memory(
                content=self._build_memory_content(question, answer),
                user_id=user_id,
                session_id=session_id,
                metadata=self._build_memory_metadata(user_id, session_id, qa_id),
            )
            self._record_memory_id(user_id, session_id, qa_id, str(memory.id))
        except Exception as e:
            logger.warning(
                "Headroom mirror failed, continuing with FS cache only: %s", e, exc_info=True
            )

    async def _delete_headroom_memories(self, memory_ids: list[str]) -> set[str]:
        """Delete each memory on its own; return the ids whose delete failed.

        One failing id must not stop the rest, and callers put the failed ids
        back into the map so a later delete or ``prune()`` retries them.
        """
        if not memory_ids:
            return set()
        try:
            backend = self._get_backend()
        except Exception as e:
            logger.warning(
                "Headroom backend unavailable, %d memories kept for a later delete: %s",
                len(memory_ids),
                e,
                exc_info=True,
            )
            return set(memory_ids)
        failed: set[str] = set()
        for memory_id in memory_ids:
            try:
                await backend.delete_memory(memory_id)
            except Exception as e:
                failed.add(memory_id)
                logger.warning(
                    "Headroom memory %s delete failed, kept for a later delete: %s",
                    memory_id,
                    e,
                    exc_info=True,
                )
        return failed

    async def _delete_mapped_memories(self, key: str, mapping: dict[str, str]) -> None:
        """Delete the memories of a popped ``qa_id -> memory_id`` map; re-record failures."""
        failed = await self._delete_headroom_memories(list(mapping.values()))
        if failed:
            self._restore_memory_ids(
                key, {qa_id: mid for qa_id, mid in mapping.items() if mid in failed}
            )

    # ------------------------------------------------------------------
    # CacheDBInterface overrides
    # ------------------------------------------------------------------

    async def create_qa_entry(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        qa_id: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
        used_graph_element_ids: dict | None = None,
        memify_metadata: dict | None = None,
        used_session_context_ids: list | None = None,
    ):
        # Generate the id here rather than reading the last entry back: another
        # process appending to the same session could otherwise slip in between.
        qa_id = qa_id or str(uuid.uuid4())
        await super().create_qa_entry(
            user_id,
            session_id,
            question,
            context,
            answer,
            qa_id,
            feedback_text,
            feedback_score,
            used_graph_element_ids=used_graph_element_ids,
            memify_metadata=memify_metadata,
            used_session_context_ids=used_session_context_ids,
        )
        await self._mirror_qa_to_headroom(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            question=question,
            answer=answer,
        )

    async def update_qa_entry(
        self,
        user_id: str,
        session_id: str,
        qa_id: str,
        question: str | None = None,
        context: str | None = None,
        answer: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
        used_graph_element_ids: dict | None = None,
        memify_metadata: dict | None = None,
        used_session_context_ids: list | None = None,
    ) -> bool:
        updated = await super().update_qa_entry(
            user_id,
            session_id,
            qa_id,
            question,
            context,
            answer,
            feedback_text,
            feedback_score,
            used_graph_element_ids=used_graph_element_ids,
            memify_metadata=memify_metadata,
            used_session_context_ids=used_session_context_ids,
        )
        if not updated or (question is None and answer is None):
            return updated

        # The mirrored text changed: replace the memory rather than leave a stale one.
        entries = await self.get_qa_entries_by_ids(user_id, session_id, [qa_id])
        if not entries:
            return updated
        old = self._pop_memory_ids(user_id, session_id, qa_id)
        failed = await self._delete_headroom_memories(list(old.values()))
        if failed:
            # The slot is about to hold the new memory; park the undeletable old
            # ones under the entry's stale keys so delete/prune still reach them.
            self._restore_memory_ids(
                self._memory_ids_key(user_id, session_id),
                {self._stale_key(qa_id, mid): mid for mid in failed},
            )
        await self._mirror_qa_to_headroom(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            question=entries[0].question,
            answer=entries[0].answer,
        )
        return updated

    async def delete_qa_entry(self, user_id: str, session_id: str, qa_id: str) -> bool:
        deleted = await super().delete_qa_entry(user_id, session_id, qa_id)
        if deleted:
            mapping = self._pop_memory_ids(user_id, session_id, qa_id)
            await self._delete_mapped_memories(self._memory_ids_key(user_id, session_id), mapping)
        return deleted

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        existed = await super().delete_session(user_id, session_id)
        mapping = self._pop_all_memory_ids(user_id, session_id)
        await self._delete_mapped_memories(self._memory_ids_key(user_id, session_id), mapping)
        return existed or bool(mapping)

    async def prune(self) -> None:
        """Empty the FS cache and remove the Headroom memories it had mirrored.

        Only memories recorded by this adapter are deleted; the Headroom store is
        shared with other agents and is never cleared wholesale. Memories whose
        delete fails keep their map entry past the prune, so the next prune or
        session delete retries them.
        """
        mirrored: dict[str, dict[str, str]] = {}
        try:
            for key in list(self.cache.iterkeys()):
                if isinstance(key, str) and key.startswith(self._MEMORY_IDS_PREFIX):
                    value = self.cache.get(key)
                    if value:
                        mirrored[key] = json.loads(value)
        except Exception as e:
            logger.warning("Could not enumerate mirrored Headroom memories: %s", e, exc_info=True)

        await super().prune()
        for key, mapping in mirrored.items():
            await self._delete_mapped_memories(key, mapping)

    async def close(self):
        if self._headroom_backend is not None:
            try:
                await self._headroom_backend.close()
            except Exception as e:
                logger.debug("Error closing Headroom backend: %s", e, exc_info=True)
            self._headroom_backend = None
        await super().close()


def _ensure_headroom_installed() -> None:
    """Fail at construction, not on the first session write, when headroom is missing."""
    if importlib.util.find_spec("headroom") is None:
        raise HeadroomNotInstalledError()
    if (
        importlib.util.find_spec("sqlite_vec") is None
        and importlib.util.find_spec("hnswlib") is None
    ):
        raise HeadroomNotInstalledError(
            "CACHE_BACKEND=headroom needs a vector index for Headroom's memory store, but "
            "neither `sqlite-vec` nor `hnswlib` is installed."
        )
