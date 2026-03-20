import asyncio
from uuid import UUID
from typing import Union, BinaryIO, List, Optional, Any

from pydantic import BaseModel

from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem

logger = get_logger("remember")


async def remember(
    data: Union[BinaryIO, list[BinaryIO], str, list[str], DataItem, list[DataItem]],
    dataset_name: str = "main_dataset",
    user: User = None,
    node_set: Optional[List[str]] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    dataset_id: Optional[UUID] = None,
    preferred_loaders: Optional[List[Union[str, dict[str, dict[str, Any]]]]] = None,
    incremental_loading: bool = True,
    data_per_batch: Optional[int] = 20,
    # cognify params
    graph_model: BaseModel = KnowledgeGraph,
    chunker=None,
    chunk_size: int = None,
    chunks_per_batch: int = None,
    run_in_background: bool = False,
    custom_prompt: Optional[str] = None,
    **kwargs,
):
    """Ingest data and build the knowledge graph in a single call.

    This is a convenience function that combines ``add()`` and ``cognify()``
    into one step.

    When ``run_in_background`` is *True* the **entire** operation (add then
    cognify) is launched as a background task and the function returns
    immediately.  When *False* (default) both steps run sequentially and the
    function blocks until they finish.

    Args:
        data: The data to ingest (text, file paths, binary streams, etc.).
        dataset_name: Target dataset. Defaults to ``"main_dataset"``.
        user: User context. Uses default user when *None*.
        node_set: Optional node identifiers for graph organisation.
        vector_db_config: Custom vector DB config.
        graph_db_config: Custom graph DB config.
        dataset_id: Explicit dataset UUID (instead of *dataset_name*).
        preferred_loaders: Custom loader configuration.
        incremental_loading: Enable incremental loading (default *True*).
        data_per_batch: Items per ingestion batch (default *20*).
        graph_model: Pydantic model for the knowledge graph structure.
        chunker: Text chunking strategy. Defaults to *TextChunker*.
        chunk_size: Max tokens per chunk. Auto-calculated when *None*.
        chunks_per_batch: Chunks per cognify batch.
        run_in_background: If *True*, run the whole remember operation
            (add + cognify) as a background task. If *False* (default),
            block until both steps complete.
        custom_prompt: Custom prompt for entity extraction.

    Returns:
        When blocking: the result of the cognify step (pipeline run info).
        When background: a dict with initial pipeline run info (returned
        immediately while the operation continues in the background).
    """
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    # Resolve chunker default here so we don't import at module level
    if chunker is None:
        from cognee.modules.chunking.TextChunker import TextChunker

        chunker = TextChunker

    if run_in_background:
        # Launch the whole add-then-cognify sequence as a background task.
        # Return immediately with a status dict so the caller isn't blocked.
        async def _remember_background():
            try:
                await add(
                    data=data,
                    dataset_name=dataset_name,
                    user=user,
                    node_set=node_set,
                    vector_db_config=vector_db_config,
                    graph_db_config=graph_db_config,
                    dataset_id=dataset_id,
                    preferred_loaders=preferred_loaders,
                    incremental_loading=incremental_loading,
                    data_per_batch=data_per_batch,
                )

                datasets_arg = [dataset_name] if dataset_id is None else [dataset_id]

                await cognify(
                    datasets=datasets_arg,
                    user=user,
                    graph_model=graph_model,
                    chunker=chunker,
                    chunk_size=chunk_size,
                    chunks_per_batch=chunks_per_batch,
                    vector_db_config=vector_db_config,
                    graph_db_config=graph_db_config,
                    run_in_background=False,  # already in a background task
                    incremental_loading=incremental_loading,
                    custom_prompt=custom_prompt,
                    **kwargs,
                )
            except Exception:
                logger.exception("Background remember failed")

        asyncio.create_task(_remember_background())

        return {
            "status": "started",
            "dataset_name": dataset_name,
            "dataset_id": str(dataset_id) if dataset_id else None,
            "run_in_background": True,
        }

    # Blocking mode: run add then cognify sequentially
    await add(
        data=data,
        dataset_name=dataset_name,
        user=user,
        node_set=node_set,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        dataset_id=dataset_id,
        preferred_loaders=preferred_loaders,
        incremental_loading=incremental_loading,
        data_per_batch=data_per_batch,
    )

    datasets_arg = [dataset_name] if dataset_id is None else [dataset_id]

    return await cognify(
        datasets=datasets_arg,
        user=user,
        graph_model=graph_model,
        chunker=chunker,
        chunk_size=chunk_size,
        chunks_per_batch=chunks_per_batch,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        run_in_background=False,
        incremental_loading=incremental_loading,
        custom_prompt=custom_prompt,
        **kwargs,
    )
