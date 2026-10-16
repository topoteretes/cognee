"""Adapter layer for interacting with Cognee either embedded or via remote API."""

import logging
from typing import Any, Dict, List, Optional

try:
    from .config import settings
    from .models import (
        AddMemoryResponse,
        PruneMemoryResponse,
        SearchMemoryResponse,
        SearchTypeEnum,
    )
except (ImportError, ValueError):
    from config import settings
    from models import (
        AddMemoryResponse,
        PruneMemoryResponse,
        SearchMemoryResponse,
        SearchTypeEnum,
    )

logger = logging.getLogger("cognee-fastmcp")


class CogneeAdapter:
    """Async adapter providing unified interface to Cognee operations."""

    def __init__(self):
        self._api_url = settings.COGNEE_API_URL
        self._api_key = settings.COGNEE_API_KEY
        self._cognee_module = None

    def _get_cognee(self):
        """Lazy load cognee module for embedded mode."""
        if self._cognee_module is None:
            try:
                import cognee
                self._cognee_module = cognee
            except ImportError as e:
                raise RuntimeError(
                    "Cognee is not installed. Please install it using 'pip install cognee' "
                    "or configure COGNEE_API_URL to use a remote Cognee instance."
                ) from e
        return self._cognee_module

    async def add_memory(
        self,
        content: str,
        dataset_name: Optional[str] = None,
        auto_cognify: bool = True
    ) -> AddMemoryResponse:
        """Ingests content and optionally extracts graph entities."""
        dataset = dataset_name or settings.DEFAULT_DATASET
        logger.info(f"Adding memory to dataset '{dataset}' (auto_cognify={auto_cognify})")

        try:
            if self._api_url:
                return await self._remote_add(content, dataset, auto_cognify)
            else:
                c = self._get_cognee()
                await c.add(data=content, dataset_name=dataset)

                cognified = False
                if auto_cognify:
                    await c.cognify(datasets=[dataset])
                    cognified = True

                return AddMemoryResponse(
                    success=True,
                    dataset_name=dataset,
                    message=f"Successfully added content to '{dataset}'." + (" Cognified graph." if cognified else ""),
                    cognified=cognified,
                    error=None
                )
        except Exception as e:
            logger.error(f"Error in add_memory: {e}", exc_info=True)
            return AddMemoryResponse(
                success=False,
                dataset_name=dataset,
                message="Failed to add memory.",
                cognified=False,
                error=str(e)
            )

    async def search_memory(
        self,
        query: str,
        search_type: SearchTypeEnum = SearchTypeEnum.INSIGHTS,
        dataset_name: Optional[str] = None
    ) -> SearchMemoryResponse:
        """Searches Cognee knowledge graph and vector store."""
        logger.info(f"Searching memory with query='{query}', type={search_type.value}, dataset={dataset_name}")

        try:
            if self._api_url:
                return await self._remote_search(query, search_type, dataset_name)
            else:
                c = self._get_cognee()

                st_val = search_type.value
                try:
                    from cognee.modules.search.types import SearchType
                    target_st = getattr(SearchType, st_val, SearchType.INSIGHTS)
                except (ImportError, AttributeError):
                    target_st = st_val

                kwargs: Dict[str, Any] = {
                    "query_text": query,
                    "search_type": target_st
                }
                if dataset_name:
                    kwargs["datasets"] = [dataset_name]

                raw_results = await c.search(**kwargs)

                results_list = raw_results if isinstance(raw_results, list) else [raw_results]
                formatted_results = [
                    res.dict() if hasattr(res, "dict") else (res.model_dump() if hasattr(res, "model_dump") else res)
                    for res in results_list
                ]

                return SearchMemoryResponse(
                    success=True,
                    query=query,
                    search_type=search_type.value,
                    dataset_name=dataset_name,
                    total_results=len(formatted_results),
                    results=formatted_results,
                    error=None
                )
        except Exception as e:
            logger.error(f"Error in search_memory: {e}", exc_info=True)
            return SearchMemoryResponse(
                success=False,
                query=query,
                search_type=search_type.value,
                dataset_name=dataset_name,
                total_results=0,
                results=[],
                error=str(e)
            )

    async def prune_memory(self, prune_all: bool = False) -> PruneMemoryResponse:
        """Prunes Cognee memory or entire system."""
        try:
            if self._api_url:
                return await self._remote_prune(prune_all)
            else:
                c = self._get_cognee()
                if prune_all:
                    await c.prune.prune_system()
                    msg = "Entire Cognee system pruned."
                else:
                    await c.prune.prune_data()
                    msg = "Cognee dataset memories pruned."
                return PruneMemoryResponse(success=True, message=msg, prune_all=prune_all)
        except Exception as e:
            logger.error(f"Error in prune_memory: {e}", exc_info=True)
            return PruneMemoryResponse(success=False, message="Prune failed.", prune_all=prune_all, error=str(e))

    async def _remote_add(self, content: str, dataset_name: str, auto_cognify: bool) -> AddMemoryResponse:
        import httpx
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with httpx.AsyncClient(base_url=self._api_url, timeout=60.0) as client:
            resp = await client.post(
                "/api/v1/add",
                json={"data": content, "dataset_name": dataset_name},
                headers=headers
            )
            resp.raise_for_status()

            if auto_cognify:
                cog_resp = await client.post(
                    "/api/v1/cognify",
                    json={"datasets": [dataset_name]},
                    headers=headers
                )
                cog_resp.raise_for_status()

        return AddMemoryResponse(
            success=True,
            dataset_name=dataset_name,
            message=f"Remote memory added to '{dataset_name}'.",
            cognified=auto_cognify
        )

    async def _remote_search(
        self,
        query: str,
        search_type: SearchTypeEnum,
        dataset_name: Optional[str]
    ) -> SearchMemoryResponse:
        import httpx
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        payload: Dict[str, Any] = {"query": query, "search_type": search_type.value}
        if dataset_name:
            payload["dataset_name"] = dataset_name

        async with httpx.AsyncClient(base_url=self._api_url, timeout=30.0) as client:
            resp = await client.post(
                "/api/v1/search",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            items = data if isinstance(data, list) else data.get("results", [data])

        return SearchMemoryResponse(
            success=True,
            query=query,
            search_type=search_type.value,
            dataset_name=dataset_name,
            total_results=len(items),
            results=items
        )

    async def _remote_prune(self, prune_all: bool) -> PruneMemoryResponse:
        import httpx
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with httpx.AsyncClient(base_url=self._api_url, timeout=30.0) as client:
            resp = await client.post(
                "/api/v1/prune",
                json={"prune_all": prune_all},
                headers=headers
            )
            resp.raise_for_status()
        return PruneMemoryResponse(success=True, message="Remote memory pruned.", prune_all=prune_all)


default_adapter = CogneeAdapter()
