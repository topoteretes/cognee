import asyncio
import time
from uuid import UUID
from typing import Union, BinaryIO, List, Optional, Any

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem
from cognee.memory import (
    MemoryEntry,
    QAEntry,
    TraceEntry,
    FeedbackEntry,
)
from cognee.memory.entries import MEMORY_ENTRY_TYPES
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.observability import (
    new_span,
    COGNEE_DATASET_NAME,
    COGNEE_SESSION_ID,
    COGNEE_DATA_SIZE_BYTES,
    COGNEE_OPERATION_MODE,
    COGNEE_DATA_ITEM_COUNT,
    OtelStatusCode,
)

logger = get_logger("remember")


class RememberKwargs(TypedDict, total=False):
    """Power-user overrides for remember(). Most users never need these."""

    graph_model: Any
    node_set: List[str]
    dataset_id: UUID
    preferred_loaders: list
    incremental_loading: bool
    data_per_batch: int
    chunks_per_batch: int
    user: object
    vector_db_config: dict
    graph_db_config: dict


# Kwarg routing: which RememberKwargs go to add(), cognify(), or both.
# Kept in sync with RememberKwargs above and the add()/cognify() signatures.
_ADD_ONLY = frozenset({"dataset_id", "node_set", "preferred_loaders", "importance_weight"})
_COGNIFY_ONLY = frozenset({"graph_model", "chunks_per_batch", "config", "temporal_cognify"})
_SHARED = frozenset(
    {
        "user",
        "vector_db_config",
        "graph_db_config",
        "incremental_loading",
        "data_per_batch",
        "run_in_background",
    }
)


def _estimate_data_size(data) -> int:
    """Estimate the byte size of input data."""
    if isinstance(data, str):
        return len(data.encode("utf-8", errors="replace"))
    if isinstance(data, bytes):
        return len(data)
    if isinstance(data, list):
        return sum(_estimate_data_size(item) for item in data)
    if hasattr(data, "seek") and hasattr(data, "tell"):
        pos = data.tell()
        data.seek(0, 2)
        size = data.tell()
        data.seek(pos)
        return size
    return 0


def _data_to_text(data) -> str:
    """Convert ingested data to its full text representation."""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "name"):
                parts.append(f"[file: {item.name}]")
            else:
                parts.append(f"[{type(item).__name__}]")
        return "\n\n".join(parts)
    if hasattr(data, "name"):
        return f"[file: {data.name}]"
    return f"[{type(data).__name__}]"


_SESSION_PLACEHOLDER_PREFIXES = ("[UploadFile]", "[file:", "[BinaryIO", "[SpooledTemporaryFile")


async def _add_to_session(session_id: str, data, user):
    """Add a Q&A entry to the session cache.

    Sessions store chat-shaped content (prompts, assistant answers,
    Q&A turns). File-upload data coerces to placeholder strings like
    ``[UploadFile]`` / ``[file: name]`` — those are useless in the
    session cache and pollute recall results, so they're skipped.
    """
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    sm = get_session_manager()
    if not sm.is_available:
        logger.warning("remember: session cache not available (enable CACHING=true)")
        return

    user_id = str(user.id) if user and hasattr(user, "id") else None
    if not user_id:
        return

    text = _data_to_text(data)
    stripped = text.strip()
    if not stripped:
        return
    if any(stripped.startswith(prefix) for prefix in _SESSION_PLACEHOLDER_PREFIXES):
        logger.debug(
            "remember: skipping session write for placeholder-only payload (%.40s…)",
            stripped,
        )
        return

    await sm.add_qa(
        user_id=user_id,
        session_id=session_id,
        question="",
        context="",
        answer=text,
    )
    logger.info("remember: added entry to session '%s'", session_id)


async def _remember_entry(
    entry,
    *,
    dataset_name: str,
    session_id: Optional[str],
    user,
) -> "RememberResult":
    """Top-level dispatcher for typed MemoryEntry payloads.

    Routes to the remote HTTP client when ``cognee.serve(url=...)`` is
    active, otherwise runs the SessionManager call in-process.
    """
    from cognee.api.v1.serve.state import get_remote_client

    client = get_remote_client()
    if client is not None:
        payload = await client.remember_entry(
            entry,
            dataset_name=dataset_name,
            session_id=session_id,
        )
        # Reconstruct a RememberResult from the server's response
        result = RememberResult(
            status=payload.get("status", "session_stored"),
            dataset_name=payload.get("dataset_name", dataset_name),
            session_ids=payload.get("session_ids"),
        )
        result.entry_type = payload.get("entry_type")
        result.entry_id = payload.get("entry_id")
        result.elapsed_seconds = payload.get("elapsed_seconds")
        if payload.get("error"):
            result.error = payload["error"]
        return result

    return await _dispatch_session_entry(
        entry,
        dataset_name=dataset_name,
        session_id=session_id,
        user=user,
    )


async def _dispatch_session_entry(
    entry: "MemoryEntry",
    *,
    dataset_name: str,
    session_id: Optional[str],
    user,
) -> "RememberResult":
    """Route a typed memory entry to the right SessionManager method.

    All typed entries require a session_id — session cache is the
    storage target. Returns a RememberResult with entry_id/entry_type
    fields populated so callers can chain (e.g., attach feedback to a
    QA they just stored).
    """
    from cognee.infrastructure.session.get_session_manager import get_session_manager
    from cognee.modules.engine.operations.setup import setup

    if not session_id:
        raise ValueError(
            f"session_id is required for typed memory entries (got {type(entry).__name__})"
        )

    await setup()

    if user is None:
        from cognee.modules.users.methods import get_default_user

        user = await get_default_user()

    user_id = str(user.id) if user and hasattr(user, "id") else None
    if not user_id:
        raise ValueError("Could not resolve user for session entry")

    sm = get_session_manager()
    if not sm.is_available:
        raise RuntimeError("Session cache unavailable — set CACHING=true to enable session memory")

    # Resolve the dataset UUID for this session so the session_records
    # row carries dataset_id — otherwise downstream permission filtering
    # (e.g. the dashboard listing) can't match the session via granted
    # dataset reads. We backfill via ensure_and_touch_session, which
    # upserts the row and only sets dataset_id when currently null.
    try:
        from uuid import UUID as _UUID

        from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset
        from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session

        resolved_dataset = None
        try:
            ds = await get_authorized_dataset(user, dataset_name, "write")
            if ds is not None:
                resolved_dataset = ds.id
        except Exception:
            # Fall through with None — we still create the session row.
            resolved_dataset = None

        await ensure_and_touch_session(
            session_id=session_id,
            user_id=_UUID(user_id),
            dataset_id=resolved_dataset,
        )
    except Exception as exc:
        logger.debug("remember: pre-upsert session_record failed (%s)", exc)

    result = RememberResult(
        status="session_stored",
        dataset_name=dataset_name,
        session_ids=[session_id],
    )
    result.elapsed_seconds = time.monotonic() - result._started_at
    result.entry_type = entry.type

    if isinstance(entry, QAEntry):
        qa_id = await sm.add_qa(
            user_id=user_id,
            session_id=session_id,
            question=entry.question,
            context=entry.context,
            answer=entry.answer,
            feedback_text=entry.feedback_text,
            feedback_score=entry.feedback_score,
            used_graph_element_ids=entry.used_graph_element_ids,
        )
        result.entry_id = qa_id
        if qa_id is None:
            result.status = "errored"
            result.error = "SessionManager.add_qa returned None"
        return result

    if isinstance(entry, TraceEntry):
        trace_id = await sm.add_agent_trace_step(
            user_id=user_id,
            session_id=session_id,
            origin_function=entry.origin_function,
            status=entry.status,
            generate_feedback_with_llm=entry.generate_feedback_with_llm,
            memory_query=entry.memory_query,
            memory_context=entry.memory_context,
            method_params=entry.method_params,
            method_return_value=entry.method_return_value,
            error_message=entry.error_message,
        )
        result.entry_id = trace_id
        if trace_id is None:
            result.status = "errored"
            result.error = "SessionManager.add_agent_trace_step returned None"
        return result

    if isinstance(entry, FeedbackEntry):
        ok = await sm.add_feedback(
            user_id=user_id,
            session_id=session_id,
            qa_id=entry.qa_id,
            feedback_text=entry.feedback_text,
            feedback_score=entry.feedback_score,
        )
        result.entry_id = entry.qa_id
        if not ok:
            result.status = "errored"
            result.error = f"add_feedback: QA {entry.qa_id} not found in session {session_id}"
        return result

    raise TypeError(f"Unsupported memory entry type: {type(entry).__name__}")


class RememberResult:
    """Promise-like result from ``remember()``.

    Can be printed for a quick summary, awaited to block until the
    pipeline finishes (background mode), or inspected via attributes.

    Attributes:
        status: ``"running"``, ``"completed"``, ``"errored"``,
                or ``"session_stored"``.
        dataset_name: Target dataset.
        dataset_id: Dataset UUID (str) when available.
        session_id: Session ID (session-only mode).
        pipeline_run_id: Pipeline run UUID (str) when available.
        error: Error message if the pipeline failed.
        elapsed_seconds: Wall-clock time from start to completion.
        content_hash: Content hash of the processed data (first item).
        items_processed: Number of data items processed.
        items: List of dicts with per-item info (name, content_hash,
            token_count) for each data item in the pipeline run.
        raw_result: The original cognify() return value (dict of
            dataset_id -> PipelineRunInfo) for advanced inspection.

    Example::

        result = await cognee.remember("Einstein was born in Ulm.")
        print(result)
        # RememberResult(status='completed', dataset='main_dataset',
        #                items=1, elapsed=4.2s)

        result.content_hash   # 'a1b2c3...'
        result.items          # [{'name': '...', 'content_hash': '...', ...}]

        # Background mode:
        result = await cognee.remember("data", run_in_background=True)
        print(result.done)   # False
        await result          # blocks until done
        print(result)        # status='completed'
    """

    def __init__(
        self,
        *,
        status: str,
        dataset_name: str,
        dataset_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
        pipeline_run_id: Optional[str] = None,
    ):
        self.status = status
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.session_ids: Optional[List[str]] = session_ids
        self.pipeline_run_id = pipeline_run_id
        self.error: Optional[str] = None
        self.raw_result: Optional[dict] = None
        self.elapsed_seconds: Optional[float] = None
        self.content_hash: Optional[str] = None
        self.items_processed: int = 0
        self.items: List[dict] = []
        # Populated when the call dispatched a typed MemoryEntry.
        # entry_type is one of "qa", "trace", "feedback"; entry_id is
        # the qa_id / trace_id returned by SessionManager (or the
        # qa_id a feedback was attached to).
        self.entry_type: Optional[str] = None
        self.entry_id: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._started_at: float = time.monotonic()

    @property
    def session_id(self) -> Optional[str]:
        """The session ID when exactly one session is involved, else None."""
        if self.session_ids and len(self.session_ids) == 1:
            return self.session_ids[0]
        return None

    def __repr__(self):
        parts = [f"status={self.status!r}", f"dataset={self.dataset_name!r}"]
        if self.session_ids:
            if len(self.session_ids) == 1:
                parts.append(f"session_id={self.session_ids[0]!r}")
            else:
                parts.append(f"session_ids={self.session_ids!r}")
        if self.dataset_id:
            parts.append(f"dataset_id={self.dataset_id!r}")
        if self.pipeline_run_id:
            parts.append(f"pipeline_run_id={self.pipeline_run_id!r}")
        if self.items_processed:
            parts.append(f"items={self.items_processed}")
        if self.content_hash:
            parts.append(f"content_hash={self.content_hash!r}")
        if self.elapsed_seconds is not None:
            parts.append(f"elapsed={self.elapsed_seconds:.1f}s")
        if self.error:
            parts.append(f"error={self.error!r}")
        return f"RememberResult({', '.join(parts)})"

    def __str__(self):
        return repr(self)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON API responses."""
        d = {
            "status": self.status,
            "dataset_name": self.dataset_name,
            "dataset_id": self.dataset_id,
            "pipeline_run_id": self.pipeline_run_id,
            "items_processed": self.items_processed,
            "elapsed_seconds": self.elapsed_seconds,
        }
        if self.session_ids:
            d["session_ids"] = self.session_ids
        if self.content_hash:
            d["content_hash"] = self.content_hash
        if self.items:
            d["items"] = self.items
        if self.entry_type:
            d["entry_type"] = self.entry_type
        if self.entry_id:
            d["entry_id"] = self.entry_id
        if self.error:
            d["error"] = self.error
        return d

    def __bool__(self):
        """True if status is completed or session_stored."""
        return self.status in ("completed", "session_stored")

    def __await__(self):
        return self._await_impl().__await__()

    async def _await_impl(self):
        """Await the background task if running, then return self.

        The background coroutine handles its own exceptions and writes
        them to ``self.error`` / ``self.status``, so this method never
        re-raises — it just waits for the task to finish.
        """
        if self._task is not None and not self._task.done():
            await asyncio.shield(self._task)
        return self

    @property
    def done(self) -> bool:
        """True if the pipeline has finished (success or failure).

        For session-stored results (no pipeline runs), always True.
        For blocking results (no background task), reflects status.
        For background results, delegates to the asyncio.Task.
        """
        if self._task is not None:
            return self._task.done()
        return self.status != "running"

    def _resolve(self, cognify_result):
        """Extract fields from the cognify() return value.

        Called once when the pipeline finishes. Sets status, dataset_id,
        pipeline_run_id, elapsed_seconds, item info, and stores the raw
        result.
        """
        self.raw_result = cognify_result
        self.elapsed_seconds = time.monotonic() - self._started_at

        if not cognify_result or not isinstance(cognify_result, dict):
            self.status = "completed"
            return

        # cognify returns {dataset_id: PipelineRunInfo}
        # remember() always processes a single dataset, so take the first.
        ds_id, run_info = next(iter(cognify_result.items()))
        self.dataset_id = str(ds_id)

        if hasattr(run_info, "status"):
            self.status = "errored" if "Errored" in run_info.status else "completed"
            if hasattr(run_info, "pipeline_run_id"):
                self.pipeline_run_id = str(run_info.pipeline_run_id)
        else:
            self.status = "completed"

        # Extract per-item details from the payload (list of Data objects)
        payload = getattr(run_info, "payload", None)
        if payload and isinstance(payload, list):
            for data_item in payload:
                item_info = {}
                if hasattr(data_item, "id"):
                    item_info["id"] = str(data_item.id)
                if hasattr(data_item, "name"):
                    item_info["name"] = data_item.name
                if hasattr(data_item, "content_hash"):
                    item_info["content_hash"] = data_item.content_hash
                if hasattr(data_item, "token_count"):
                    item_info["token_count"] = data_item.token_count
                if hasattr(data_item, "mime_type"):
                    item_info["mime_type"] = data_item.mime_type
                if hasattr(data_item, "data_size"):
                    item_info["data_size"] = data_item.data_size
                if item_info:
                    self.items.append(item_info)

            self.items_processed = len(self.items)
            if self.items and self.items[0].get("content_hash"):
                self.content_hash = self.items[0]["content_hash"]

    def _fail(self, exc: BaseException):
        """Mark the result as failed with an error message and elapsed time."""
        self.status = "errored"
        self.error = str(exc)
        self.elapsed_seconds = time.monotonic() - self._started_at


async def remember(
    data: Union[
        BinaryIO,
        list[BinaryIO],
        str,
        list[str],
        DataItem,
        list[DataItem],
        "MemoryEntry",
    ],
    dataset_name: str = "main_dataset",
    *,
    session_id: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunker: Optional[Any] = None,
    custom_prompt: Optional[str] = None,
    run_in_background: bool = False,
    self_improvement: bool = True,
    session_ids: Optional[List[str]] = None,
    **kwargs: Unpack[RememberKwargs],
) -> "RememberResult":
    """Store data in memory.

    Two modes depending on whether ``session_id`` is provided:

    **Without session_id (permanent memory):** Runs ``add()`` +
    ``cognify()`` to ingest data and build the knowledge graph.

    **With session_id (session memory):** Stores the data in the
    session cache for fast retrieval. When ``self_improvement`` is
    True (default), also bridges the session data into the permanent
    graph in the background via ``improve()``. The call returns
    immediately — await the result to wait for the background sync.

    Args:
        data: The data to store (text, file paths, binary streams, etc.).
        dataset_name: Target dataset. Defaults to ``"main_dataset"``.
        session_id: Optional session ID. When set, stores data in the
            session cache instead of the permanent graph.
        chunk_size: Max tokens per chunk. Auto-calculated when *None*.
        chunker: Text chunking strategy. Defaults to *TextChunker*.
        custom_prompt: Custom prompt for entity extraction.
        run_in_background: If *True*, run as a background task.
        self_improvement: If *True* (default), automatically runs
            ``improve()`` after cognify to enrich the graph with
            triplet embeddings and indexing.
        session_ids: Session IDs to sync graph knowledge back to.
            Only used when ``self_improvement=True``. When provided,
            ``improve()`` will also copy recent graph relationships
            into these sessions for fast retrieval.
        **kwargs: Additional options -- see ``RememberKwargs``.

    Returns:
        RememberResult: A promise-like object. Print it for a summary,
        await it to block until background processing finishes, or
        inspect ``.status``, ``.dataset_name``, ``.elapsed_seconds``, etc.

    Example::

        result = await cognee.remember("Einstein was born in Ulm.")
        print(result)
        # RememberResult(status='completed', dataset='main_dataset', elapsed=4.2s)

        # Background mode:
        result = await cognee.remember("data", run_in_background=True)
        print(result)        # status='running'
        await result          # blocks until done
        print(result)        # status='completed'

        # Access raw pipeline result:
        result.raw_result    # {dataset_id: PipelineRunInfo}
    """
    from cognee.shared.utils import send_telemetry
    from cognee import __version__ as cognee_version

    # Typed MemoryEntry dispatch: trace steps, rich QA, feedback.
    # These short-circuit the add+cognify path entirely and write
    # directly into the session cache via SessionManager.
    if isinstance(data, MEMORY_ENTRY_TYPES):
        return await _remember_entry(
            data,
            dataset_name=dataset_name,
            session_id=session_id,
            user=kwargs.get("user"),
        )

    data_size = _estimate_data_size(data)
    item_count = len(data) if isinstance(data, list) else 1
    mode = "session" if session_id else "permanent"

    with new_span("cognee.api.remember") as span:
        span.set_attribute(COGNEE_DATASET_NAME, dataset_name)
        span.set_attribute(COGNEE_OPERATION_MODE, mode)
        span.set_attribute(COGNEE_DATA_SIZE_BYTES, data_size)
        span.set_attribute(COGNEE_DATA_ITEM_COUNT, item_count)
        if session_id:
            span.set_attribute(COGNEE_SESSION_ID, session_id)

        send_telemetry(
            "cognee.remember",
            kwargs.get("user", "sdk"),
            additional_properties={
                "mode": mode,
                "dataset_name": dataset_name,
                "data_size_bytes": data_size,
                "item_count": item_count,
                "session_id": session_id or "",
                "self_improvement": self_improvement,
                "run_in_background": run_in_background,
                "cognee_version": cognee_version,
            },
        )

        return await _remember_inner(
            data,
            dataset_name,
            session_id=session_id,
            chunk_size=chunk_size,
            chunker=chunker,
            custom_prompt=custom_prompt,
            run_in_background=run_in_background,
            self_improvement=self_improvement,
            session_ids=session_ids,
            span=span,
            **kwargs,
        )


async def _remember_inner(
    data,
    dataset_name,
    *,
    session_id,
    chunk_size,
    chunker,
    custom_prompt,
    run_in_background,
    self_improvement,
    session_ids,
    span,
    **kwargs,
) -> "RememberResult":
    from cognee.api.v1.serve.state import get_remote_client

    client = get_remote_client()
    if client is not None:
        span.set_attribute(COGNEE_OPERATION_MODE, "cloud")
        return await client.remember(data, dataset_name, session_id=session_id, **kwargs)

    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    if chunker is None:
        from cognee.modules.chunking.TextChunker import TextChunker

        chunker = TextChunker

    # Route kwargs to add(), cognify(), or both
    remaining = dict(kwargs)
    add_kwargs = {}
    cognify_kwargs = {}
    shared_kwargs = {}

    for key in list(remaining):
        if key in _SHARED:
            shared_kwargs[key] = remaining.pop(key)
        elif key in _ADD_ONLY:
            add_kwargs[key] = remaining.pop(key)
        elif key in _COGNIFY_ONLY:
            cognify_kwargs[key] = remaining.pop(key)

    if remaining:
        raise TypeError(f"Unexpected keyword arguments: {', '.join(remaining)}")

    dataset_id = add_kwargs.pop("dataset_id", None) or shared_kwargs.get("dataset_id")

    # Ensure database is initialized (same as add() does internally).
    # Must run before get_default_user() which queries the DB.
    from cognee.modules.engine.operations.setup import setup

    await setup()

    # Resolve user early so we can use it for session init
    user = shared_kwargs.get("user")
    if user is None:
        from cognee.modules.users.methods import get_default_user

        user = await get_default_user()
        shared_kwargs["user"] = user

    # Session memory: store in session cache, then optionally bridge to graph
    if session_id:
        await _add_to_session(session_id, data, user)
        result = RememberResult(
            status="session_stored",
            dataset_name=dataset_name,
            session_ids=[session_id],
        )
        result.elapsed_seconds = time.monotonic() - result._started_at

        # Bridge session data to permanent graph in the background
        if self_improvement:
            from cognee.api.v1.improve import improve

            async def _session_improve():
                try:
                    await improve(
                        dataset=dataset_name,
                        session_ids=[session_id],
                        user=user,
                    )
                    logger.info("remember: session '%s' bridged to permanent graph", session_id)
                except Exception as exc:
                    logger.warning("remember: session improve failed (non-fatal): %s", exc)

            result._task = asyncio.create_task(_session_improve())

        return result

    # Build the result object — starts as "running"
    if not dataset_id and dataset_name:
        # Create dataset if it doesn't exist
        user, dataset_id = await resolve_authorized_user_datasets(dataset_name, user)
        dataset_id = dataset_id[0].id if dataset_id else None
    result = RememberResult(
        status="running",
        dataset_name=dataset_name,
        dataset_id=str(dataset_id) if dataset_id else None,
        session_ids=session_ids,
    )

    # Permanent memory: add + cognify (+ optional improve)
    async def _run():
        await add(
            data=data,
            dataset_name=dataset_name,
            **shared_kwargs,
            **add_kwargs,
        )

        datasets_arg = [dataset_name] if dataset_id is None else [dataset_id]

        cognify_result = await cognify(
            datasets=datasets_arg,
            chunker=chunker,
            chunk_size=chunk_size,
            custom_prompt=custom_prompt,
            run_in_background=False,
            **shared_kwargs,
            **cognify_kwargs,
        )

        result._resolve(cognify_result)

        if self_improvement:
            from cognee.api.v1.improve import improve

            logger.info("remember: running self-improvement on dataset '%s'", dataset_name)
            improve_kwargs = {"dataset": dataset_name, "user": user}
            if session_ids:
                improve_kwargs["session_ids"] = session_ids
            await improve(**improve_kwargs)

    if run_in_background:

        async def _remember_background():
            try:
                await _run()
            except Exception as exc:
                result._fail(exc)
                logger.exception("Background remember failed")

        result._task = asyncio.create_task(_remember_background())
        return result

    # Blocking mode
    try:
        await _run()
    except Exception as exc:
        result._fail(exc)
        raise

    return result
