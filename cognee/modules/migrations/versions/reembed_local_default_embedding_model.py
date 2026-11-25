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

"Configured" is what the installation has configured. A model one call asks for
(``set_database_global_context_variables(..., embedding_config=...)``) is not in
force here, also when that call is what started the migrations.

A re-embed that stopped half-way is finished by the next run. When a model was
configured in between, a re-embed that had replaced nothing yet is undone, so the
dataset keeps the previous model; one that had started replacing collections raises,
because it can only be finished with the local default.

With backend access control off there is no dataset to re-embed: every dataset shares
one vector store. A shared store that holds vectors of the previous default raises,
with the setting that keeps the previous model.

A failed re-embed is raised with what gets the dataset out of it: that setting when
the dataset is as it was, another run when it is partly re-embedded. A failed
migration blocks writes to the dataset.

Unlike the id migrations this one cannot be a frozen snapshot: its target is the
default of the release that runs it. What it moves datasets away from is frozen.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        MODEL_KEY,
        REEMBED_KEY,
    )
    from cognee.modules.data.methods.reembed_dataset import abandon_reembed_in_context

    with _installation_models():
        default_model = _local_default_in_force()
        if default_model == _PREVIOUS_MODEL:
            return
        if context.dataset_id is None:
            if default_model is not None:
                await _refuse_shared_store_of_previous_default(context.vector_engine, default_model)
            return

        record = await _embedding_record(context.dataset_id)
        if record.get(MODEL_KEY) != _PREVIOUS_MODEL:
            if default_model is not None and record.get(MODEL_KEY) is None:
                await _warn_of_unnamed_previous_width(context, record, default_model)
            return
        if default_model is None:
            # A model is configured, so the dataset stays as it is. Not half-way,
            # though: a dataset marked as being re-embedded refuses every use.
            if REEMBED_KEY in record:
                await abandon_reembed_in_context(context.dataset_id)
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
        await _reembed(context.dataset_id, default_model)


async def downgrade(context: MigrationContext) -> None:
    """Re-embed the dataset with the previous default, if the local default built it.

    For going back to a release whose local default is the previous model. The
    dataset routing collection keeps its width: that release rebuilds it after a
    ``prune_dataset_routes``.
    """
    from cognee.context_global_variables import DatabaseContextManager
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import MODEL_KEY
    from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
    from cognee.modules.data.methods.reembed_dataset import reembed_dataset_in_context

    with _installation_models():
        default_model = _local_default_in_force()
        if context.dataset_id is None or default_model in (None, _PREVIOUS_MODEL):
            return
        if (await _embedding_record(context.dataset_id)).get(MODEL_KEY) != default_model:
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


@contextmanager
def _installation_models() -> Iterator[None]:
    """Put the installation's own LLM and embedding settings in force.

    Migrations also start from inside an operation (a failed run is retried by the
    next write), and that operation may run with models of its own. Seen from in
    there the local default would look configured away, and the dataset would be
    stamped as migrated without having been touched.
    """
    from cognee.context_global_variables import embedding_config, llm_config

    llm_token = llm_config.set(None)
    embedding_token = embedding_config.set(None)
    try:
        yield
    finally:
        embedding_config.reset(embedding_token)
        llm_config.reset(llm_token)


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


async def _embedding_record(dataset_id: UUID) -> dict:
    """What the dataset's registry row says about its vectors: the model that built
    them and, while a re-embed is unfinished, that."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import DatasetDatabase

    async with get_relational_engine().get_async_session() as session:
        info = await session.scalar(
            select(DatasetDatabase.vector_database_connection_info).where(
                DatasetDatabase.dataset_id == dataset_id
            )
        )
    return info or {}


async def _warn_of_unnamed_previous_width(
    context: MigrationContext, record: dict, default_model: str
) -> None:
    """Say that a dataset is left alone whose vectors have the previous default's
    width while its record names no model: which model built it is not known."""
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import DIMENSIONS_KEY

    stored_dimensions = record.get(DIMENSIONS_KEY)
    if stored_dimensions is None:
        read_stored_size = getattr(context.vector_engine, "get_stored_vector_size", None)
        if read_stored_size is not None:
            stored_dimensions = await read_stored_size()
    if stored_dimensions != _PREVIOUS_DIMENSIONS:
        return
    logger.warning(
        "Dataset %s holds %d-dimensional vectors and its record does not name the model "
        "that built them (it was built before cognee 1.6.0), so it is not re-embedded "
        "with the new default local model %s and cannot be used with it. If %s built "
        "it: %s Or re-embed it with reembed_dataset (cognee.modules.data.methods).",
        context.dataset_id,
        _PREVIOUS_DIMENSIONS,
        default_model,
        _PREVIOUS_MODEL,
        _KEEP_PREVIOUS_MODEL,
    )


async def _reembed(dataset_id: UUID, model: str) -> None:
    """Re-embed the dataset with the local default; a failure says what gets out of it."""
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        MODEL_KEY,
        REEMBED_KEY,
    )
    from cognee.modules.data.methods.reembed_dataset import reembed_dataset_in_context

    try:
        # Loads the model, so a model that cannot be fetched fails in here too.
        await _drop_routes_of_another_width()
        reembedded = await reembed_dataset_in_context(dataset_id)
    except Exception as error:
        record = await _embedding_record(dataset_id)
        if record.get(MODEL_KEY) == _PREVIOUS_MODEL and REEMBED_KEY not in record:
            state = f"was not re-embedded with '{model}' and is as it was. {_KEEP_PREVIOUS_MODEL}"
        else:
            state = (
                f"is partly re-embedded with '{model}'. Run the migrations again "
                "(cognee-cli upgrade): they continue where this stopped."
            )
        # The runner keeps the first 500 characters of this for `cognee-cli current`,
        # so the way out comes before the cause.
        raise CogneeSystemError(message=f"Dataset {dataset_id} {state} Cause: {error}") from error
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
