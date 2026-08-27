"""Fallback for graph content that a memory-only forget cannot see via provenance.

``remember(content_type="code")`` writes its graph nodes and edges through a
payload that carries no persisted data-item id (the payload is a repository
path string — see ``add_data_points``), so neither the relational rollback
ledger nor graph-provenance marking ever covers them.
``delete_dataset_nodes_and_edges`` (whichever branch it takes) can then find
nothing to remove for a dataset that is entirely code, or clear the document
half of a dataset that mixes cognified documents with code and leave the code
half behind — which matters because both ``remember(text)`` and
``remember(content_type="code")`` default to ``dataset_name="main_dataset"``,
so a mixed dataset is a likely default, not an edge case.

This module runs after that provenance-driven step and answers two questions:
is the dataset's graph actually empty now, and — if not — can the rest be
cleared safely. "Safely" means the dataset owns its own isolated graph and
vector database on a backend whose per-dataset store is a plain file that the
next connection recreates on its own — not a server-managed database/schema
that a DROP leaves gone until something re-issues CREATE, which nothing does
today (see ``get_or_create_dataset_database``, which reuses an existing
registry row without ever calling the handler's ``create_dataset`` again).

Isolation is what makes a reset safe even for a mixed dataset: each dataset's
store is one physical database, named after the dataset id (e.g.
``f"{dataset_id}.lbug"``), so nothing belonging to another dataset could ever
have been written into it. The provenance-driven step already removed (or,
for content shared across two datasets, correctly detached) everything it
could attribute to THIS dataset — cross-dataset sharing cannot exist in an
isolated store, so there is nothing it could have "kept" that isn't this
dataset's own. Whatever is still in the store afterward is this dataset's own
unprovenanced content, and clearing it is exactly what ``memory_only``
promises — regardless of whether the provenance step found anything for this
dataset or not. That is why the decision below no longer looks at what the
provenance step found; only the isolation and self-healing-backend gates
control it.

Databases that are NOT isolated (``ENABLE_BACKEND_ACCESS_CONTROL=false``) get
no reset — nothing here is safe to wipe when other datasets could share the
same store — but they still get one narrow, cheap, read-only honesty check:
see ``_UNPROVENANCED_PIPELINE_NAME`` below.
"""

from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.utils.delete_isolated_dataset_storage import (
    delete_isolated_dataset_storage,
)
from cognee.infrastructure.databases.utils.get_or_create_dataset_database import (
    get_existing_dataset_database,
)
from cognee.modules.pipelines.methods import get_pipeline_run_by_dataset
from cognee.shared.logging_utils import get_logger

logger = get_logger("ensure_graph_memory_cleared")

# Embedded, file-based per-dataset stores: dropping the file is self-healing,
# because the next connection recreates it. Server-managed per-dataset
# databases/schemas (Neo4j, the postgres_demo graph, PGVector) are
# deliberately excluded — their handlers issue DROP DATABASE / DROP SCHEMA,
# and nothing re-issues CREATE before the next write.
_SELF_HEALING_DATASET_DATABASE_HANDLERS = frozenset(
    {"ladybug", "kuzu", "lancedb", "turso_graph", "turso"}
)

# The one pipeline that writes graph content with no deletion provenance today
# (see cognee/tasks/code_graph/extract_code_graph.py, which every
# remember(content_type="code") call runs under this pipeline_name).
# run_tasks calls log_pipeline_run_start unconditionally at the top of every
# run — use_pipeline_cache (always False for this pipeline; see
# run_pipeline_per_dataset) only gates whether an existing PipelineRun row is
# later *read* to skip re-processing, never whether one is *written* — so a
# row existing for this dataset is a reliable, cheap, index-backed signal
# (dataset_id, pipeline_name is a covered index) that such content was
# written at some point. It is not a certainty: an errored run could exist
# with no content ever having been written.
_UNPROVENANCED_PIPELINE_NAME = "code_graph_pipeline"


@dataclass
class GraphMemoryStatus:
    """Whether a dataset's graph is confirmed free of memory-relevant content.

    ``cleared`` is only ``True`` when this call is confident nothing
    memory-relevant remains. ``note`` explains why when it is not, or how the
    dataset ended up cleared when a fallback reset was needed.
    """

    cleared: bool
    note: Optional[str] = None


async def ensure_graph_memory_cleared(dataset_id: UUID, user: Any) -> GraphMemoryStatus:
    """Confirm a dataset's graph is empty; reset it as a last resort if safe.

    Call this after the provenance-driven delete (``delete_dataset_nodes_and_edges``)
    already ran — it never widens or pre-empts that step, only inspects (and, when
    safe, finishes clearing) the graph afterward. On any unexpected error this
    returns ``cleared=False`` rather than raising, so a forget() call stays safe
    to retry.
    """
    try:
        if not backend_access_control_enabled():
            code_run = await get_pipeline_run_by_dataset(dataset_id, _UNPROVENANCED_PIPELINE_NAME)
            if code_run is not None:
                return GraphMemoryStatus(
                    cleared=False,
                    note=(
                        "This dataset has a remember(content_type='code') run on "
                        "record, so unprovenanced content may remain — this "
                        "deployment shares graph/vector databases across datasets "
                        "(ENABLE_BACKEND_ACCESS_CONTROL=false), so completeness "
                        "could not be verified."
                    ),
                )
            return GraphMemoryStatus(cleared=True)

        dataset_database = await get_existing_dataset_database(dataset_id, user)
        if dataset_database is None:
            # No isolated store was ever provisioned for this dataset.
            return GraphMemoryStatus(cleared=True)

        # _forget_dataset_memory always calls this from inside
        # set_database_global_context_variables(resolved_dataset_id, user.id),
        # which sets the graph_db_config ContextVar to THIS dataset's own
        # connection info before get_graph_engine() is ever reached here (see
        # DatabaseContextManager.apply_database_context_variables). So this
        # resolves — and, via the process-wide engine cache, is keyed on —
        # this dataset's own isolated engine, never another dataset's or the
        # process-wide default. get_graph_context_config() is what reads that
        # ContextVar; get_graph_engine() builds its cache key from it.
        graph_engine = await get_graph_engine()
        if await graph_engine.is_empty():
            return GraphMemoryStatus(cleared=True)

        # Non-empty. See the module docstring: in an isolated store, nothing
        # left after the provenance-driven step ran can belong to any other
        # dataset, so it is safe to remove regardless of what that step found.
        graph_handler_name = dataset_database.graph_dataset_database_handler
        vector_handler_name = dataset_database.vector_dataset_database_handler
        if (
            graph_handler_name not in _SELF_HEALING_DATASET_DATABASE_HANDLERS
            or vector_handler_name not in _SELF_HEALING_DATASET_DATABASE_HANDLERS
        ):
            return GraphMemoryStatus(
                cleared=False,
                note=(
                    "This dataset's graph is not empty, but a full per-dataset "
                    f"store reset is not yet supported for the '{graph_handler_name}'/"
                    f"'{vector_handler_name}' backend, so its remaining content "
                    "(e.g. from remember(content_type='code')) was left in place."
                ),
            )

        await delete_isolated_dataset_storage(dataset_database)
        logger.info(
            "forget: reset isolated graph+vector store for dataset=%s "
            "(content had no deletion provenance)",
            dataset_id,
        )
        return GraphMemoryStatus(
            cleared=True,
            note=(
                "Cleared by resetting this dataset's isolated graph and vector "
                "store: its remaining content had no deletion provenance to "
                "remove it by (e.g. it was ingested via remember(content_type="
                "'code'))."
            ),
        )
    except Exception as error:
        logger.warning(
            "forget: could not verify graph memory was cleared for dataset=%s (non-fatal): %s",
            dataset_id,
            error,
        )
        return GraphMemoryStatus(
            cleared=False,
            note="Could not verify whether this dataset's graph memory was fully cleared.",
        )
