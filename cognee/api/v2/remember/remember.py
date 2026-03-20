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

    graph_model: Any  # Pydantic model for knowledge graph structure
    node_set: List[str]  # Node identifiers for graph organization
    dataset_id: UUID  # Explicit dataset UUID instead of name
    preferred_loaders: list  # Custom loader configuration
    incremental_loading: bool  # Enable incremental loading (default True)
    data_per_batch: int  # Items per ingestion batch (default 20)
    chunks_per_batch: int  # Chunks per cognify batch
    user: object  # User context (resolved internally when None)
    vector_db_config: dict  # Custom vector DB config (multi-tenant)
    graph_db_config: dict  # Custom graph DB config (multi-tenant)


def _build_param_sets():
    """Pre-compute which kwargs belong to add() vs cognify() vs both."""
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    add_params = frozenset(inspect.signature(add).parameters) - {"data", "dataset_name"}
    cognify_params = frozenset(inspect.signature(cognify).parameters) - {"datasets"}
    shared = add_params & cognify_params
    return add_params, cognify_params, shared


# Lazily initialized on first call to avoid import-time side effects.
_ADD_PARAMS: Optional[frozenset] = None
_COGNIFY_PARAMS: Optional[frozenset] = None
_SHARED_PARAMS: Optional[frozenset] = None


def _ensure_param_sets():
    global _ADD_PARAMS, _COGNIFY_PARAMS, _SHARED_PARAMS
    if _ADD_PARAMS is None:
        _ADD_PARAMS, _COGNIFY_PARAMS, _SHARED_PARAMS = _build_param_sets()


async def remember(
    data: Union[BinaryIO, list[BinaryIO], str, list[str], DataItem, list[DataItem]],
    dataset_name: str = "main_dataset",
    *,
    chunk_size: Optional[int] = None,
    chunker: Optional[Any] = None,
    custom_prompt: Optional[str] = None,
    run_in_background: bool = False,
    **kwargs: Unpack[RememberKwargs],
):
    """Ingest data and build the knowledge graph in a single call.

    This is a convenience function that combines ``add()`` and ``cognify()``
    into one step.  The most common parameters are explicit keyword arguments;
    power-user options can be passed via ``RememberKwargs``.

    When ``run_in_background`` is *True* the **entire** operation (add then
    cognify) is launched as a background task and the function returns
    immediately.

    Args:
        data: The data to ingest (text, file paths, binary streams, etc.).
        dataset_name: Target dataset. Defaults to ``"main_dataset"``.
        chunk_size: Max tokens per chunk. Auto-calculated when *None*.
        chunker: Text chunking strategy. Defaults to *TextChunker*.
        custom_prompt: Custom prompt for entity extraction.
        run_in_background: If *True*, run as a background task.
        **kwargs: Additional options — see ``RememberKwargs``.

    Returns:
        When blocking: the result of the cognify step (pipeline run info).
        When background: a dict with initial pipeline run info.
    """
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    _ensure_param_sets()

    # Resolve chunker default
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
