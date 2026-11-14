"""Dataset routing: pick the dataset(s) an unscoped search covers, from the global context index.

``route.py`` is the read path (``authorized_search``), ``index.py`` the write
path (the global context index build, dataset deletion, prune), ``store.py``
where the rows live and ``config.py`` the settings. ``candidates.py`` filters
the readable datasets to the ones the configured embedding model can score.
Runs only with backend access control on.
"""

from .candidates import datasets_on_configured_model
from .config import DatasetRoutingConfig, get_dataset_routing_config
from .index import (
    dataset_route_ids,
    prune_dataset_routes,
    remove_dataset_routes,
    sync_dataset_routes,
)
from .route import DatasetRoutingDecision, route_datasets
from .store import (
    DATASET_ROUTE_COLLECTION,
    dataset_route_collection,
    dataset_route_collection_name,
    ensure_default_store,
    get_dataset_route_engine,
    route_model_record,
)

__all__ = [
    "DATASET_ROUTE_COLLECTION",
    "DatasetRoutingConfig",
    "DatasetRoutingDecision",
    "dataset_route_collection",
    "dataset_route_collection_name",
    "dataset_route_ids",
    "datasets_on_configured_model",
    "ensure_default_store",
    "get_dataset_route_engine",
    "get_dataset_routing_config",
    "prune_dataset_routes",
    "remove_dataset_routes",
    "route_datasets",
    "route_model_record",
    "sync_dataset_routes",
]
