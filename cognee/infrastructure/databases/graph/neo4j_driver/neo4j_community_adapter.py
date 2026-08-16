from typing import Any, Optional

from .adapter import Neo4jAdapter
from .neo4j_community_containers import get_container_manager


class Neo4jCommunityAdapter(Neo4jAdapter):
    """Neo4j adapter bound to a per-dataset Docker container.

    Selected by ``_create_graph_engine`` when the graph dataset database
    handler is ``neo4j_community``. The only behavioral difference from the
    plain adapter is teardown: engine instances live in the
    ``closing_lru_cache``, which calls ``close()`` once an entry is evicted and
    no caller holds its lease — at that point the dataset is idle, so the
    backing container is stopped as well (auto-scale-down). The next operation
    on the dataset restarts it via
    ``Neo4jCommunityDatasetDatabaseHandler.resolve_dataset_connection_info``.
    """

    def __init__(
        self,
        graph_database_url: str,
        graph_database_username: Optional[str] = None,
        graph_database_password: Optional[str] = None,
        graph_database_name: Optional[str] = None,
        graph_database_allow_anonymous: bool = False,
        driver: Optional[Any] = None,
    ):
        super().__init__(
            graph_database_url=graph_database_url,
            graph_database_username=graph_database_username,
            graph_database_password=graph_database_password,
            graph_database_name=graph_database_name,
            graph_database_allow_anonymous=graph_database_allow_anonymous,
            driver=driver,
        )
        self._community_bolt_url = graph_database_url

    async def close(self) -> None:
        await super().close()
        await get_container_manager().stop_container_for_url(self._community_bolt_url)
