"""
Cognee Client abstraction that supports both direct function calls and HTTP API calls.

This module provides a unified interface for interacting with Cognee, supporting:
- Direct mode: Directly imports and calls cognee functions (default behavior)
- API mode: Makes HTTP requests to a running Cognee FastAPI server
"""

import sys
from typing import Optional, Any, List, Dict
from uuid import UUID
from contextlib import redirect_stdout
import httpx
from cognee.shared.logging_utils import get_logger
import json

logger = get_logger()


class CogneeClient:
    """
    Unified client for interacting with Cognee via direct calls or HTTP API.

    Parameters
    ----------
    api_url : str, optional
        Base URL of the Cognee API server (e.g., "http://localhost:8000").
        If None, uses direct cognee function calls.
    api_token : str, optional
        Authentication token for the API (optional, required if API has authentication enabled).
    """

    def __init__(self, api_url: Optional[str] = None, api_token: Optional[str] = None):
        self.api_url = api_url.rstrip("/") if api_url else None
        self.api_token = api_token
        self.use_api = bool(api_url)

        # Extract tenant ID from tenant URL pattern: tenant-<uuid>.*.cognee.ai
        self.tenant_id: Optional[str] = None
        if self.api_url:
            import re

            match = re.search(r"tenant-([0-9a-f-]{36})", self.api_url)
            if match:
                self.tenant_id = match.group(1)

        if self.use_api:
            logger.info(f"Cognee client initialized in API mode: {self.api_url}")
            if self.tenant_id:
                logger.info(f"Tenant ID extracted from URL: {self.tenant_id}")
            self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout for long operations
        else:
            logger.info("Cognee client initialized in direct mode")
            # Import cognee only if we're using direct mode
            import cognee as _cognee

            self.cognee = _cognee

    def _get_headers(self, include_content_type: bool = True) -> Dict[str, str]:
        """Get headers for API requests.

        Uses X-Api-Key + X-Tenant-Id for tenant APIs (cloud),
        falls back to Bearer token for local/self-hosted backends.
        """
        headers: Dict[str, str] = {}
        if include_content_type:
            headers["Content-Type"] = "application/json"
        if self.api_token:
            if self.tenant_id:
                headers["X-Api-Key"] = self.api_token
                headers["X-Tenant-Id"] = self.tenant_id
            else:
                headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    async def add(
        self, data: Any, dataset_name: str = "main_dataset", node_set: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add data to Cognee for processing.

        Parameters
        ----------
        data : Any
            Data to add (text, file path, etc.)
        dataset_name : str
            Name of the dataset to add data to
        node_set : List[str], optional
            List of node identifiers for graph organization

        Returns
        -------
        Dict[str, Any]
            Result of the add operation
        """
        if self.use_api:
            endpoint = f"{self.api_url}/api/v1/add"

            files = {"data": ("data.txt", str(data), "text/plain")}
            form_data = {
                "datasetName": dataset_name,
            }
            if node_set is not None:
                form_data["node_set"] = json.dumps(node_set)

            response = await self.client.post(
                endpoint,
                files=files,
                data=form_data,
                headers=self._get_headers(include_content_type=False),
            )
            response.raise_for_status()
            return response.json()
        else:
            with redirect_stdout(sys.stderr):
                await self.cognee.add(data, dataset_name=dataset_name, node_set=node_set)
                return {"status": "success", "message": "Data added successfully"}

    async def cognify(
        self,
        datasets: Optional[List[str]] = None,
        custom_prompt: Optional[str] = None,
        graph_model: Any = None,
    ) -> Dict[str, Any]:
        """
        Transform data into a knowledge graph.

        Parameters
        ----------
        datasets : List[str], optional
            List of dataset names to process
        custom_prompt : str, optional
            Custom prompt for entity extraction
        graph_model : Any, optional
            Custom graph model (only used in direct mode)

        Returns
        -------
        Dict[str, Any]
            Result of the cognify operation
        """
        if self.use_api:
            # API mode: Make HTTP request
            endpoint = f"{self.api_url}/api/v1/cognify"
            payload = {
                "datasets": datasets or ["main_dataset"],
                "run_in_background": False,
            }
            if custom_prompt:
                payload["custom_prompt"] = custom_prompt

            response = await self.client.post(endpoint, json=payload, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        else:
            # Direct mode: Call cognee directly
            with redirect_stdout(sys.stderr):
                kwargs = {}
                if datasets:
                    kwargs["datasets"] = datasets
                if custom_prompt:
                    kwargs["custom_prompt"] = custom_prompt
                if graph_model:
                    kwargs["graph_model"] = graph_model

                await self.cognee.cognify(**kwargs)
                return {"status": "success", "message": "Cognify completed successfully"}

    async def search(
        self,
        query_text: str,
        query_type: str,
        datasets: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        top_k: int = 10,
    ) -> Any:
        """
        Search the knowledge graph.

        Parameters
        ----------
        query_text : str
            The search query
        query_type : str
            Type of search (e.g., "GRAPH_COMPLETION", "INSIGHTS", etc.)
        datasets : List[str], optional
            List of datasets to search
        system_prompt : str, optional
            System prompt for completion searches
        top_k : int
            Maximum number of results

        Returns
        -------
        Any
            Search results
        """
        if self.use_api:
            # API mode: Make HTTP request
            endpoint = f"{self.api_url}/api/v1/search"
            payload = {"query": query_text, "search_type": query_type.upper(), "top_k": top_k}
            if datasets:
                payload["datasets"] = datasets
            if system_prompt:
                payload["system_prompt"] = system_prompt

            response = await self.client.post(endpoint, json=payload, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        else:
            # Direct mode: Call cognee directly
            from cognee.modules.search.types import SearchType

            with redirect_stdout(sys.stderr):
                search_kwargs = {
                    "query_type": SearchType[query_type.upper()],
                    "query_text": query_text,
                    "top_k": top_k,
                }
                if datasets:
                    search_kwargs["datasets"] = datasets
                if system_prompt:
                    search_kwargs["system_prompt"] = system_prompt
                results = await self.cognee.search(**search_kwargs)
                return results

    async def delete(self, data_id: UUID, dataset_id: UUID, mode: str = "soft") -> Dict[str, Any]:
        """
        Delete data from a dataset.

        Parameters
        ----------
        data_id : UUID
            ID of the data to delete
        dataset_id : UUID
            ID of the dataset containing the data

        Returns
        -------
        Dict[str, Any]
            Result of the deletion
        """
        if self.use_api:
            # API mode: Make HTTP request
            endpoint = f"{self.api_url}/api/v1/datasets/{str(dataset_id)}/data/{str(data_id)}"

            response = await self.client.delete(endpoint, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        else:
            # Direct mode: Call cognee directly
            from cognee.modules.users.methods import get_default_user

            with redirect_stdout(sys.stderr):
                user = await get_default_user()
                await self.cognee.datasets.delete_data(
                    dataset_id=dataset_id,
                    data_id=data_id,
                    user=user,
                )

    async def prune_data(self) -> Dict[str, Any]:
        """
        Prune all data from the knowledge graph.

        Returns
        -------
        Dict[str, Any]
            Result of the prune operation
        """
        if self.use_api:
            # Note: The API doesn't expose a prune endpoint, so we'll need to handle this
            # For now, raise an error
            raise NotImplementedError("Prune operation is not available via API")
        else:
            # Direct mode: Call cognee directly
            with redirect_stdout(sys.stderr):
                await self.cognee.prune.prune_data()
                return {"status": "success", "message": "Data pruned successfully"}

    async def prune_system(self, metadata: bool = True) -> Dict[str, Any]:
        """
        Prune system data from the knowledge graph.

        Parameters
        ----------
        metadata : bool
            Whether to prune metadata

        Returns
        -------
        Dict[str, Any]
            Result of the prune operation
        """
        if self.use_api:
            # Note: The API doesn't expose a prune endpoint
            raise NotImplementedError("Prune system operation is not available via API")
        else:
            # Direct mode: Call cognee directly
            with redirect_stdout(sys.stderr):
                await self.cognee.prune.prune_system(metadata=metadata)
                return {"status": "success", "message": "System pruned successfully"}

    async def get_pipeline_status(
        self, dataset_ids: List[UUID], pipeline_name: str
    ) -> Dict[str, Any]:
        """
        Get the status of a pipeline run.

        Parameters
        ----------
        dataset_ids : List[UUID]
            List of dataset IDs
        pipeline_name : str
            Name of the pipeline

        Returns
        -------
        Dict[str, Any]
            Status information keyed by dataset ID
        """
        if self.use_api:
            # Note: This would need a custom endpoint on the API side
            raise NotImplementedError("Pipeline status is not available via API")
        else:
            # Direct mode: Call cognee directly
            from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status

            with redirect_stdout(sys.stderr):
                status = await get_pipeline_status(dataset_ids, pipeline_name)
                return status

    async def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets.

        Returns
        -------
        List[Dict[str, Any]]
            List of datasets
        """
        if self.use_api:
            # API mode: Make HTTP request
            endpoint = f"{self.api_url}/api/v1/datasets"
            response = await self.client.get(endpoint, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        else:
            # Direct mode: Call cognee directly
            from cognee.modules.users.methods import get_default_user
            from cognee.modules.data.methods import get_datasets

            with redirect_stdout(sys.stderr):
                user = await get_default_user()
                datasets = await get_datasets(user.id)
                return [
                    {"id": str(d.id), "name": d.name, "created_at": str(d.created_at)}
                    for d in datasets
                ]

    # -- V2 API methods -----------------------------------------------------

    async def remember(
        self,
        data: Any,
        dataset_name: str = "main_dataset",
        session_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store data in memory via remember().

        With session_id: stores in session cache only (fast).
        Without session_id: full add + cognify pipeline (permanent).
        """
        if self.use_api:
            endpoint = f"{self.api_url}/api/v1/remember"
            files = {"data": ("data.txt", str(data), "text/plain")}
            form_data = {"datasetName": dataset_name}
            if custom_prompt:
                form_data["custom_prompt"] = custom_prompt
            response = await self.client.post(
                endpoint,
                files=files,
                data=form_data,
                headers=self._get_headers(include_content_type=False),
            )
            response.raise_for_status()
            return response.json()
        else:
            with redirect_stdout(sys.stderr):
                kwargs = {
                    "data": data,
                    "dataset_name": dataset_name,
                }
                if session_id:
                    kwargs["session_id"] = session_id
                if custom_prompt:
                    kwargs["custom_prompt"] = custom_prompt
                result = await self.cognee.remember(**kwargs)
                return {
                    "status": getattr(result, "status", "completed"),
                    "dataset_name": dataset_name,
                    "session_id": session_id,
                }

    async def recall(
        self,
        query_text: str,
        search_type: Optional[str] = None,
        datasets: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        top_k: int = 10,
    ) -> Any:
        """Search memory via recall() with auto-routing and session awareness."""
        if self.use_api:
            endpoint = f"{self.api_url}/api/v1/recall"
            payload = {"query": query_text, "top_k": top_k}
            if search_type:
                payload["search_type"] = search_type.upper()
            if datasets:
                payload["datasets"] = datasets
            response = await self.client.post(endpoint, json=payload, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        else:
            with redirect_stdout(sys.stderr):
                kwargs = {"top_k": top_k, "auto_route": True}
                if search_type:
                    from cognee.modules.search.types import SearchType

                    kwargs["query_type"] = SearchType[search_type.upper()]
                if datasets:
                    kwargs["datasets"] = datasets
                if session_id:
                    kwargs["session_id"] = session_id
                return await self.cognee.recall(query_text=query_text, **kwargs)

    async def forget(
        self,
        dataset: Optional[str] = None,
        everything: bool = False,
    ) -> Dict[str, Any]:
        """Delete data via forget()."""
        if self.use_api:
            endpoint = f"{self.api_url}/api/v1/forget"
            payload = {"everything": everything}
            if dataset:
                payload["dataset"] = dataset
            response = await self.client.post(endpoint, json=payload, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        else:
            with redirect_stdout(sys.stderr):
                return await self.cognee.forget(dataset=dataset, everything=everything)

    async def improve(
        self,
        dataset_name: str = "main_dataset",
        session_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Enrich knowledge graph and bridge session data via improve()."""
        if self.use_api:
            endpoint = f"{self.api_url}/api/v1/improve"
            payload = {"dataset_name": dataset_name}
            if session_ids:
                payload["session_ids"] = session_ids
            response = await self.client.post(endpoint, json=payload, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        else:
            with redirect_stdout(sys.stderr):
                kwargs = {"dataset": dataset_name}
                if session_ids:
                    kwargs["session_ids"] = session_ids
                result = await self.cognee.improve(**kwargs)
                return {"status": "success", "result": str(result)}

    async def close(self):
        """Close the HTTP client if in API mode."""
        if self.use_api and hasattr(self, "client"):
            await self.client.aclose()
