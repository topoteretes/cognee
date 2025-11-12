"""Lightweight async client that talks to the Cognee REST API."""

from typing import Any, Dict, List, Optional

import httpx

from cognee.shared.logging_utils import get_logger

logger = get_logger()


class CogneeClient:
    """HTTP client wrapper used by the MCP tools."""

    def __init__(self, api_url: str, api_token: Optional[str] = None) -> None:
        if not api_url:
            raise ValueError("api_url is required")

        self.api_url = api_url.rstrip("/")
        self.api_token = api_token
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"

            self._client = httpx.AsyncClient(base_url=self.api_url, headers=headers, timeout=120.0)
            logger.debug("Created Cognee HTTP client for %s", self.api_url)

        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def list_datasets(self) -> List[Dict[str, Any]]:
        client = await self._get_client()
        response = await client.get("/api/v1/datasets")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):  # pragma: no cover - API contract
            raise ValueError("Unexpected datasets payload")
        return data

    async def search(
        self,
        *,
        query_text: str,
        datasets: Optional[List[str]],
        dataset_ids: Optional[List[str]] = None,
        search_type: str,
        top_k: int,
        system_prompt: Optional[str],
        use_combined_context: bool = False,
        only_context: bool = False,
        node_name: Optional[List[str]] = None,
    ) -> Any:
        client = await self._get_client()
        payload: Dict[str, Any] = {
            "query": query_text,
            "search_type": search_type,
            "top_k": top_k,
            "use_combined_context": use_combined_context,
            "only_context": only_context,
        }
        if datasets:
            payload["datasets"] = datasets
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if node_name:
            payload["node_name"] = node_name

        response = await client.post("/api/v1/search", json=payload)
        response.raise_for_status()
        return response.json()
