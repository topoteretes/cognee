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
see ``_UNPROVENANCED_PIPELINE_NAMES`` below.

An empty graph is not proof the isolated store is fully clear, either:
``remember(content_type="code", index_vectors=True)`` writes real vector
embeddings alongside the graph nodes, and the code-graph pipeline's own
stale-content sweep (``_sweep_stale_code_graph``) only removes graph nodes
and edges — it never touches the vector side. So a dataset can end up with an
empty graph and a non-empty vector store (most plausibly after an earlier
reset that reached the graph but not the vector store, or any other path that
empties the graph without the vector store also emptying). The same
``_UNPROVENANCED_PIPELINE_NAMES`` signal used for the non-isolated honesty
check doubles as the cheap, read-only trigger for that case below: an
otherwise-empty isolated graph with a code-graph run on record still routes
into the same reset path, so the vector store gets wiped too.
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
from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements
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

# Pipelines that write graph content with no deletion provenance today (see
# cognee/tasks/code_graph/extract_code_graph.py, which every
# remember(content_type="code") call runs under pipeline_name
# "code_graph_pipeline"). A frozenset (not a single constant) so a future
# unprovenanced pipeline is a one-line addition here — otherwise it would
# silently miss detection on non-isolated deployments and skip the
# empty-graph-but-maybe-not-vector-empty check on isolated ones.
#
# run_tasks calls log_pipeline_run_start unconditionally at the top of every
# run — use_pipeline_cache (always False for this pipeline; see
# run_pipeline_per_dataset) only gates whether an existing PipelineRun row is
# later *read* to skip re-processing, never whether one is *written* — so a
# row existing for this dataset is a reliable, cheap, index-backed signal
# (dataset_id, pipeline_name is a covered index) that such content was
# written at some point. It is not a certainty: an errored run could exist
# with no content ever having been written.
_UNPROVENANCED_PIPELINE_NAMES = frozenset({"code_graph_pipeline"})


async def _unprovenanced_pipeline_run(dataset_id: UUID):
    """First on-record PipelineRun for a pipeline in ``_UNPROVENANCED_PIPELINE_NAMES``, or None."""
    for pipeline_name in _UNPROVENANCED_PIPELINE_NAMES:
        run = await get_pipeline_run_by_dataset(dataset_id, pipeline_name)
        if run is not None:
            return run
    return None


@dataclass
class GraphMemoryStatus:
    """Whether a dataset's graph is confirmed free of memory-relevant content.

    ``cleared`` is only ``True`` when this call is confident nothing
    memory-relevant remains. ``note`` explains why when it is not, or how the
    dataset ended up cleared when a fallback reset was needed. ``deleted_elements``
    carries the node/edge identities a fallback reset physically removed (read
    from the graph immediately before the reset, since nothing survives the
    reset to read them from afterward) — ``None`` when no reset ran. Callers
    fold this into their own provenance-driven delete result so reported
    counts and session invalidation both see what the reset removed.
    """

    cleared: bool
    note: Optional[str] = None
    deleted_elements: Optional[DeletedGraphElements] = None


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
            code_run = await _unprovenanced_pipeline_run(dataset_id)
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
        graph_empty = await graph_engine.is_empty()

        if graph_empty:
            # See the module docstring: an empty graph alone does not prove
            # the vector store is empty too. Only route into a reset here
            # when the cheap unprovenanced-pipeline signal says content that
            # bypasses deletion provenance was ever written for this dataset.
            code_run = await _unprovenanced_pipeline_run(dataset_id)
            if code_run is None:
                return GraphMemoryStatus(cleared=True)
            reset_reason = (
                "its graph reads empty, but a remember(content_type='code') run "
                "is on record and any vector embeddings it wrote "
                "(index_vectors=True) are never swept by the code-graph "
                "pipeline's own stale-content sweep"
            )
        else:
            reset_reason = "its graph is not empty"

        # Non-empty (or empty-but-unprovenanced, see above). See the module
        # docstring: in an isolated store, nothing left after the
        # provenance-driven step ran can belong to any other dataset, so it
        # is safe to remove regardless of what that step found.
        graph_handler_name = dataset_database.graph_dataset_database_handler
        vector_handler_name = dataset_database.vector_dataset_database_handler
        if (
            graph_handler_name not in _SELF_HEALING_DATASET_DATABASE_HANDLERS
            or vector_handler_name not in _SELF_HEALING_DATASET_DATABASE_HANDLERS
        ):
            return GraphMemoryStatus(
                cleared=False,
                note=(
                    f"This dataset needs a full per-dataset store reset ({reset_reason}), "
                    "but that is not yet supported for the "
                    f"'{graph_handler_name}'/'{vector_handler_name}' backend, so its "
                    "remaining content (e.g. from remember(content_type='code')) was "
                    "left in place."
                ),
            )

        # Capture what the reset is about to remove BEFORE wiping the store —
        # nothing survives the reset to read this from afterward. Callers use
        # this to report accurate nodes_deleted/edges_deleted and to widen
        # session invalidation to dataset-unattributed sessions that used this
        # content (see forget.py's _forget_dataset_memory).
        nodes, edges = await graph_engine.get_graph_data()
        deleted_elements = DeletedGraphElements(
            node_ids={str(node_id) for node_id, _properties in nodes},
            edge_ids={
                str(generate_edge_object_id(str(source_id), str(target_id), relationship_name))
                for source_id, target_id, relationship_name, _properties in edges
            },
        )

        await delete_isolated_dataset_storage(dataset_database)

        # The reset just wiped whatever the CODE search snapshot cache may
        # have parsed from this dataset's graph — drop it too, or a
        # SearchType.CODE query inside the cache's TTL can still return
        # results from the store that was just reset. Local import mirrors
        # extract_code_graph.py's own invalidation call (avoids importing the
        # retrieval layer at module load time).
        from cognee.modules.retrieval.code_retriever import (
            invalidate_code_graph_snapshot_cache,
        )

        invalidate_code_graph_snapshot_cache(dataset_id=dataset_id)

        logger.info(
            "forget: reset isolated graph+vector store for dataset=%s "
            "(content had no deletion provenance)",
            dataset_id,
        )
        return GraphMemoryStatus(
            cleared=True,
            note=(
                "Cleared by resetting this dataset's isolated graph and vector "
                f"store ({reset_reason}): remaining content had no deletion "
                "provenance to remove it by (e.g. it was ingested via "
                "remember(content_type='code'))."
            ),
            deleted_elements=deleted_elements,
        )
    except Exception as error:
        logger.warning(
            "forget: could not verify graph memory was cleared for dataset=%s (non-fatal): %s",
            dataset_id,
            error,
            exc_info=True,
        )
        return GraphMemoryStatus(
            cleared=False,
            note="Could not verify whether this dataset's graph memory was fully cleared.",
        )
