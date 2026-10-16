"""Source-aware retrieval for embedded SDKs and clients connected by cognee.serve()."""

from typing import Any
from uuid import UUID


async def search(
    query: str,
    *,
    source_hint: str | None = None,
    dataset_ids: list[UUID] | None = None,
    include_connections: bool = True,
    top_k: int = 10,
    max_sources: int = 6,
    max_catalog_entries: int = 512,
    exclude_source_ids: list[UUID] | None = None,
    user: Any = None,
) -> dict:
    """Discover likely sources and use their native retrieval capabilities.

    Database tools are opt-in at deployment level and permissioned separately
    from datasets. Supplying dataset_ids restricts retrieval to those datasets.
    In served mode the API key determines identity; a local User cannot override it.
    """
    from cognee.api.v1.datasets.routers.source_routes import SearchSources
    from cognee.api.v1.serve.state import get_remote_client
    from cognee.modules.data.source_search import search_sources
    from cognee.modules.users.methods import get_default_user

    payload = SearchSources(
        query=query,
        source_hint=source_hint,
        dataset_ids=dataset_ids,
        include_connections=include_connections,
        top_k=top_k,
        max_sources=max_sources,
        max_catalog_entries=max_catalog_entries,
        exclude_source_ids=exclude_source_ids,
    )
    remote = get_remote_client()
    if remote is not None:
        if user is not None:
            raise ValueError("Served source search uses the API key principal; omit user.")
        return await remote.search_sources(payload.model_dump(mode="json"))
    return await search_sources(
        user=user if user is not None else await get_default_user(), **payload.model_dump()
    )
