"""Dataset routing: pick the dataset(s) an unscoped search covers, from the global context index.

``route.py`` is the read path (``authorized_search``), ``index.py`` the write
path (the global context index build, dataset deletion, prune), ``store.py``
where the rows live and ``config.py`` the settings. Runs only with backend
access control on.
"""

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
    DatasetRoute,
    ensure_default_store,
    get_dataset_route_engine,
    routing_width_matches,
)

__all__ = [
    "DATASET_ROUTE_COLLECTION",
    "DatasetRoute",
    "DatasetRoutingConfig",
    "DatasetRoutingDecision",
    "dataset_route_ids",
    "ensure_default_store",
    "get_dataset_route_engine",
    "get_dataset_routing_config",
    "prune_dataset_routes",
    "remove_dataset_routes",
    "route_datasets",
    "routing_width_matches",
    "sync_dataset_routes",
]
