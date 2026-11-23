"""Re-embed the datasets built with the previous default local embedding model.

With no embedding setting and no usable LLM key, embeddings run on a local default
model. That default changed from ``BAAI/bge-small-en-v1.5`` (384 dimensions) to
``perplexity-ai/pplx-embed-v1-0.6b`` (1024 dimensions), which should provide better
accuracy. A collection's vector width is fixed when it is created, so a dataset built
with the previous default can be neither written to nor searched with the new one
until its vectors are computed again, from the text stored next to them.

Runs only where the local default is in force, which is where nothing about
embeddings is configured. A configured model is used as it is and its datasets are
left alone, whatever built them. And only a dataset recorded as built with the
previous default is re-embedded (``reembed_dataset_in_context``); every other one is
left alone too, so a second run changes nothing and a new deployment does nothing.

With backend access control off there is no dataset to re-embed: every dataset shares
one vector store. A shared store that holds vectors of the previous default raises,
with the setting that keeps the previous model.

A failed re-embed is raised with that setting as well: a failed migration blocks
writes to the dataset, and with the previous model configured the next run skips it.

Unlike the id migrations this one cannot be a frozen snapshot: its target is the
default of the release that runs it. What it moves datasets away from is frozen.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from cognee.exceptions import CogneeConfigurationError, CogneeSystemError
from cognee.modules.migrations.migration import MigrationContext
from cognee.shared.logging_utils import get_logger

logger = get_logger("migration.reembed_local_default_embedding_model")

_PREVIOUS_PROVIDER = "fastembed"
_PREVIOUS_MODEL = "BAAI/bge-small-en-v1.5"
_PREVIOUS_DIMENSIONS = 384

_KEEP_PREVIOUS_MODEL = (
    f"To keep the previous model, set EMBEDDING_PROVIDER={_PREVIOUS_PROVIDER} and "
    f"EMBEDDING_MODEL={_PREVIOUS_MODEL}."
)


async def migrate(context: MigrationContext) -> None:
    """Re-embed the dataset with the local default, if the previous default built it."""
    default_model = _local_default_in_force()
    if default_model is None or default_model == _PREVIOUS_MODEL:
        return
    if context.dataset_id is None:
        await _refuse_shared_store_of_previous_default(context.vector_engine, default_model)
        return
    if await _recorded_model(context.dataset_id) != _PREVIOUS_MODEL:
        return

    logger.warning(
        "The default local embedding model changed from %s to %s, which should provide "
        "better accuracy. Dataset %s was built with the previous default and is now "
        "re-embedded with the new one. This can take a while, and the model (about "
        "0.7 GB) is downloaded on first use. %s",
        _PREVIOUS_MODEL,
        default_model,
        context.dataset_id,
        _KEEP_PREVIOUS_MODEL,
    )
    await _drop_routes_of_another_width()
    await _reembed(context.dataset_id, default_model)


async def downgrade(context: MigrationContext) -> None:
    """Re-embed the dataset with the previous default, if the local default built it.

    For going back to a release whose local default is the previous model. The
    dataset routing collection keeps its width: that release rebuilds it after a
    ``prune_dataset_routes``.
    """
    from cognee.context_global_variables import DatabaseContextManager
    from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
    from cognee.modules.data.methods.reembed_dataset import reembed_dataset_in_context

    default_model = _local_default_in_force()
    if context.dataset_id is None or default_model in (None, _PREVIOUS_MODEL):
        return
    if await _recorded_model(context.dataset_id) != default_model:
        return

    logger.warning(
        "Dataset %s is re-embedded with %s, the default local embedding model before %s.",
        context.dataset_id,
        _PREVIOUS_MODEL,
        default_model,
    )
    previous = EmbeddingConfig(
        embedding_provider=_PREVIOUS_PROVIDER,
        embedding_model=_PREVIOUS_MODEL,
        embedding_dimensions=_PREVIOUS_DIMENSIONS,
    )
    # Inside the runner's context for the same dataset: only the model changes.
    async with DatabaseContextManager(
        context.dataset_id, embedding_config=previous, check_embedding_model=False
    ):
        await reembed_dataset_in_context(context.dataset_id)


def _local_default_in_force() -> str | None:
    """The default local embedding model, when embeddings run on it; else None."""
    from cognee.infrastructure.databases.vector.embeddings.config import (
        DEFAULT_LOCAL_EMBEDDING_MODEL,
        get_embedding_context_config,
        keyless_embedding_defaults_apply,
    )
    from cognee.infrastructure.llm.config import get_llm_context_config

    if keyless_embedding_defaults_apply(get_embedding_context_config(), get_llm_context_config()):
        return DEFAULT_LOCAL_EMBEDDING_MODEL
    return None


async def _recorded_model(dataset_id: UUID) -> str | None:
    """The embedding model the dataset's registry row says built it, if it says."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import MODEL_KEY
    from cognee.modules.users.models import DatasetDatabase

    async with get_relational_engine().get_async_session() as session:
        info = await session.scalar(
            select(DatasetDatabase.vector_database_connection_info).where(
                DatasetDatabase.dataset_id == dataset_id
            )
        )
    return (info or {}).get(MODEL_KEY)


async def _reembed(dataset_id: UUID, model: str) -> None:
    """Re-embed the dataset with the local default; a failure says how to do without."""
    from cognee.modules.data.methods.reembed_dataset import reembed_dataset_in_context

    try:
        reembedded = await reembed_dataset_in_context(dataset_id)
    except Exception as error:
        # The runner keeps the first 500 characters of this for `cognee-cli current`,
        # so the way out comes before the cause.
        raise CogneeSystemError(
            message=(
                f"Dataset {dataset_id} was not re-embedded with the default local "
                f"embedding model '{model}'. {_KEEP_PREVIOUS_MODEL} Cause: {error}"
            )
        ) from error
    logger.info(
        "Dataset %s is now embedded with %s: %d row(s) in %d collection(s).",
        dataset_id,
        model,
        reembedded.rows_embedded,
        len(reembedded.collections),
    )


async def _drop_routes_of_another_width() -> None:
    """Drop the dataset routing collection if it holds vectors of another width.

    Every dataset's routes share that collection and its width is fixed too. Once
    dropped, each re-embedded dataset writes its routes to it again, at the new width.
    """
    from cognee.modules.search.dataset_routing import (
        get_dataset_route_engine,
        prune_dataset_routes,
        routing_width_matches,
    )

    if not await routing_width_matches(await get_dataset_route_engine()):
        await prune_dataset_routes()
        logger.info("Dropped the dataset routing collection: it is rebuilt at the new width.")


async def _refuse_shared_store_of_previous_default(vector_engine, default_model: str) -> None:
    """Raise if the one store every dataset shares holds vectors of the previous default."""
    read_stored_size = getattr(vector_engine, "get_stored_vector_size", None)
    if read_stored_size is None or await read_stored_size() != _PREVIOUS_DIMENSIONS:
        return
    # Short enough for the runner to keep all of it (`cognee-cli current`).
    raise CogneeConfigurationError(
        message=(
            f"The default local embedding model changed from '{_PREVIOUS_MODEL}' to "
            f"'{default_model}'. The shared vector store (backend access control is off) "
            "holds vectors of the previous one and is not re-embedded."
        ),
        remediation=(
            f"{_KEEP_PREVIOUS_MODEL} Or delete the stored data (cognee.prune) and ingest it again."
        ),
    )
