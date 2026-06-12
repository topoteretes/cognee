from typing import List, Optional
from uuid import NAMESPACE_OID, UUID, uuid5
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from cognee.modules.graph.models import Node
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.databases.relational.with_async_session import with_async_session
from cognee.modules.graph.methods.sanitize_relational_payload import sanitize_relational_payload

UPSERT_BATCH_SIZE = 1000


# When ``session`` is passed by the caller this function does NOT commit —
# the caller is responsible for committing. When no session is provided,
# ``@with_async_session`` opens one and commits it.
@with_async_session
async def upsert_nodes(
    nodes: List[DataPoint],
    tenant_id: UUID,
    user_id: UUID,
    dataset_id: UUID,
    data_id: UUID,
    session: AsyncSession,
    pipeline_run_id: Optional[UUID] = None,
):
    """
    Adds nodes to the nodes table.

    Parameters:
    -----------
        - nodes (list): A list of nodes to be added to the graph.
    """
    if not nodes:
        return

    node_rows = [
        {
            "id": uuid5(
                NAMESPACE_OID,
                str(tenant_id) + str(user_id) + str(dataset_id) + str(data_id) + str(node.id),
            ),
            "slug": node.id,
            "user_id": user_id,
            "data_id": data_id,
            "dataset_id": dataset_id,
            "pipeline_run_id": pipeline_run_id,
            "type": sanitize_relational_payload(node.type),
            "indexed_fields": sanitize_relational_payload(
                DataPoint.get_embeddable_property_names(node)
            ),
            "label": sanitize_relational_payload(
                getattr(node, "label", getattr(node, "name", str(node.id)))
            ),
            "attributes": sanitize_relational_payload(jsonable_encoder(node)),
        }
        for node in nodes
    ]

    for start_index in range(0, len(node_rows), UPSERT_BATCH_SIZE):
        node_batch = node_rows[start_index : start_index + UPSERT_BATCH_SIZE]
        # on_conflict_do_nothing intentionally preserves the FIRST run's
        # pipeline_run_id on a re-cognify. The ledger id is keyed by logical
        # identity (tenant/user/dataset/data_id/node) and NOT by run, so a single
        # row tracks the run that originally created the node. Overwriting it with
        # a later run's id would make that later run's rollback delete a node that
        # an earlier (successful) run created — i.e. destroy pre-existing data.
        # Keeping the original tag means rollback only removes artifacts the run
        # actually introduced. (Re-writes of an existing node's attributes by a
        # later run are not separately rolled back; restoring those would require
        # per-run snapshots, which is out of scope.)
        upsert_statement = (
            insert(Node).values(node_batch).on_conflict_do_nothing(index_elements=["id"])
        )
        await session.execute(upsert_statement)
