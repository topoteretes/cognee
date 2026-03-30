import asyncio
import inspect
from uuid import UUID
from typing import Union, BinaryIO, List, Optional, Any

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem

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


def _build_param_sets():
    """Pre-compute which kwargs belong to add() vs cognify() vs both."""
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    add_params = frozenset(inspect.signature(add).parameters) - {"data", "dataset_name"}
    cognify_params = frozenset(inspect.signature(cognify).parameters) - {"datasets"}
    shared = add_params & cognify_params
    return add_params, cognify_params, shared


_ADD_PARAMS: Optional[frozenset] = None
_COGNIFY_PARAMS: Optional[frozenset] = None
_SHARED_PARAMS: Optional[frozenset] = None


def _ensure_param_sets():
    global _ADD_PARAMS, _COGNIFY_PARAMS, _SHARED_PARAMS
    if _ADD_PARAMS is None:
        _ADD_PARAMS, _COGNIFY_PARAMS, _SHARED_PARAMS = _build_param_sets()


def _summarize_data(data) -> str:
    """Extract a text summary from the ingested data for session logging."""
    if isinstance(data, str):
        return data[:500]
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, str):
                parts.append(item[:200])
            elif hasattr(item, "name"):
                parts.append(f"[file: {item.name}]")
            else:
                parts.append(f"[{type(item).__name__}]")
        return ", ".join(parts)[:500]
    if hasattr(data, "name"):
        return f"[file: {data.name}]"
    return f"[{type(data).__name__}]"


async def _init_session(session_id: str, data, dataset_name: str, user):
    """Record the remember() call as the first entry in a session."""
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    sm = get_session_manager()
    if not sm.is_available:
        return

    user_id = str(user.id) if user and hasattr(user, "id") else None
    if not user_id:
        return

    data_summary = _summarize_data(data)

    await sm.add_qa(
        user_id=user_id,
        session_id=session_id,
        question=f"[User provided data to dataset '{dataset_name}']",
        context="",
        answer=data_summary,
    )
    logger.info("remember: initialized session '%s' with ingested data summary", session_id)


async def remember(
    data: Union[BinaryIO, list[BinaryIO], str, list[str], DataItem, list[DataItem]],
    dataset_name: str = "main_dataset",
    *,
    session_id: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunker: Optional[Any] = None,
    custom_prompt: Optional[str] = None,
    run_in_background: bool = False,
    **kwargs: Unpack[RememberKwargs],
):
    """Store data in memory.

    Two modes depending on whether ``session_id`` is provided:

    **Without session_id (permanent memory):** Runs ``add()`` +
    ``cognify()`` to ingest data and build the knowledge graph.

    **With session_id (session memory):** Stores the data in the
    session cache only. The data is NOT ingested into the permanent
    graph. Use ``improve(session_ids=[...])`` later to sync session
    content into the permanent graph.

    Args:
        data: The data to store (text, file paths, binary streams, etc.).
        dataset_name: Target dataset. Defaults to ``"main_dataset"``.
        session_id: Optional session ID. When set, stores data in the
            session cache instead of the permanent graph.
        chunk_size: Max tokens per chunk. Auto-calculated when *None*.
        chunker: Text chunking strategy. Defaults to *TextChunker*.
        custom_prompt: Custom prompt for entity extraction.
        run_in_background: If *True*, run as a background task.
        **kwargs: Additional options -- see ``RememberKwargs``.

    Returns:
        When session_id: dict with session status.
        When blocking: the result of the cognify step (pipeline run info).
        When background: a dict with initial pipeline run info.
    """
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    _ensure_param_sets()

    if chunker is None:
        from cognee.modules.chunking.TextChunker import TextChunker

        chunker = TextChunker

    # Route kwargs to add(), cognify(), or both
    remaining = dict(kwargs)
    add_kwargs = {}
    cognify_kwargs = {}
    shared_kwargs = {}

    for key in list(remaining):
        if key in _SHARED_PARAMS:
            shared_kwargs[key] = remaining.pop(key)
        elif key in _ADD_PARAMS:
            add_kwargs[key] = remaining.pop(key)
        elif key in _COGNIFY_PARAMS:
            cognify_kwargs[key] = remaining.pop(key)

    if remaining:
        raise TypeError(f"Unexpected keyword arguments: {', '.join(remaining)}")

    dataset_id = add_kwargs.pop("dataset_id", None) or shared_kwargs.get("dataset_id")

    # Resolve user early so we can use it for session init
    user = shared_kwargs.get("user")
    if user is None:
        from cognee.modules.users.methods import get_default_user

        user = await get_default_user()
        shared_kwargs["user"] = user

    # Session memory: store in session cache only, skip permanent graph
    if session_id:
        await _init_session(session_id, data, dataset_name, user)
        return {
            "status": "session_stored",
            "session_id": session_id,
            "dataset_name": dataset_name,
        }

    # Permanent memory: add + cognify into the knowledge graph
    async def _run():
        await add(
            data=data,
            dataset_name=dataset_name,
            **shared_kwargs,
            **add_kwargs,
        )

        datasets_arg = [dataset_name] if dataset_id is None else [dataset_id]

        return await cognify(
            datasets=datasets_arg,
            chunker=chunker,
            chunk_size=chunk_size,
            custom_prompt=custom_prompt,
            run_in_background=False,
            **shared_kwargs,
            **cognify_kwargs,
        )

    if run_in_background:

        async def _remember_background():
            try:
                await _run()
            except Exception:
                logger.exception("Background remember failed")

        asyncio.create_task(_remember_background())

        return {
            "status": "started",
            "dataset_name": dataset_name,
            "dataset_id": str(dataset_id) if dataset_id else None,
            "run_in_background": True,
        }

    return await _run()
