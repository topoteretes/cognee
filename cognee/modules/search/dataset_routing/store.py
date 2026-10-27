"""The routing collection: one vector store for the whole deployment.

Dataset routing scores a query against every readable dataset before any
dataset is searched, so its rows cannot live inside a dataset's own vector
store. They live in the deployment's default vector store instead, the one
``VECTOR_DB_URL`` configures and every per-dataset database derives from,
which stands unused while backend access control binds each operation to its
dataset's store. The engine is built from the global config, never from the
per-dataset ContextVar, so an upsert from inside a dataset's pipeline run and
a read from search reach the same collection.
"""

from __future__ import annotations

from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.vector.get_vector_engine import _VectorEngineHandle
from cognee.infrastructure.engine import DataPoint

DATASET_ROUTE_COLLECTION = "DatasetRoute_text"


class DatasetRoute(DataPoint):
    """One routing row: a global context index summary, tagged with its dataset.

    It keeps the id of the ``GlobalContextSummary`` it mirrors, so the read path
    derives every row id it may score from the dataset and document ids alone.
    """

    text: str
    dataset_id: str
    is_root: bool = False
    metadata: dict = {"index_fields": ["text"]}


def get_dataset_route_engine() -> _VectorEngineHandle:
    """The vector engine of the deployment's default store, outside any dataset context.

    A re-resolving handle, like ``get_vector_engine_async`` returns: the routing
    writes hold it across several awaits while dataset pipelines open and evict
    engines from the shared cache, and a bare adapter reference would then be
    closed under them.
    """
    return _VectorEngineHandle(get_vectordb_config().to_dict())
