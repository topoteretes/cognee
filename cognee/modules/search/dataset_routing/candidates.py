"""Which readable datasets the configured embedding model can route between.

A dataset's registry row records the model and width that built it (written
by ``get_or_create_dataset_database``, guarded per entry by
``ensure_embedding_model_matches``). The rows are read here, from the
relational database alone -- no vector store is opened. A dataset recorded
with another model is not a candidate: it is not scored, not searched and not
widened into, because entering it would fail on the model guard anyway and
its vectors must never be ranked against another model's query. A dataset
with no recorded model, or one recorded before models were (its model reads
``None``), is a candidate when its recorded width matches: the width is what
a collection fixes and what the model guard compares, and the dataset gets
its model recorded on its first use.
"""

from __future__ import annotations

from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("dataset_routing")


async def datasets_on_configured_model(dataset_ids: list[UUID]) -> set[UUID]:
    """The datasets whose recorded embedding model is the configured one.

    A registry read that fails keeps every dataset, the searchable set routing
    started from: a candidate filter must never make a search worse, and
    entering the dataset of another model then fails the way it did before
    routing.
    """
    if not dataset_ids:
        return set()
    try:
        return await _read_candidates(dataset_ids)
    except Exception:
        logger.warning(
            "Dataset routing could not read the recorded embedding models; "
            "every readable dataset stays searchable.",
            exc_info=True,
        )
        return set(dataset_ids)


async def _read_candidates(dataset_ids: list[UUID]) -> set[UUID]:
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        DIMENSIONS_KEY,
        MODEL_KEY,
    )
    from cognee.modules.users.models import DatasetDatabase

    from .store import route_model_record

    record = route_model_record()
    configured_model = record[MODEL_KEY]
    configured_dimensions = record[DIMENSIONS_KEY]

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = (
            await session.scalars(
                select(DatasetDatabase).where(DatasetDatabase.dataset_id.in_(dataset_ids))
            )
        ).all()
    info_by_id = {row.dataset_id: (row.vector_database_connection_info or {}) for row in rows}

    candidates = set()
    for dataset_id in dataset_ids:
        info = info_by_id.get(dataset_id)
        if info is None:
            # No registry row yet: the dataset has never been through a dataset
            # context, so it holds no vectors and is always searched.
            candidates.add(dataset_id)
            continue
        model = info.get(MODEL_KEY)
        dimensions = info.get(DIMENSIONS_KEY)
        if model is None:
            # A legacy row records the width that matters, not the model.
            if dimensions in (None, configured_dimensions):
                candidates.add(dataset_id)
        elif model == configured_model and dimensions == configured_dimensions:
            candidates.add(dataset_id)
    return candidates
