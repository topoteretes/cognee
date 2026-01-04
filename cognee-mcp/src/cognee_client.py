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

        if self.use_api:
            logger.info(f"Cognee client initialized in API mode: {self.api_url}")
            self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout for long operations
        else:
            logger.info("Cognee client initialized in direct mode")
            # Import cognee only if we're using direct mode
            import cognee as _cognee

            self.cognee = _cognee

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_token:
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
                headers={"Authorization": f"Bearer {self.api_token}"} if self.api_token else {},
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
        top_k: int = 5,
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
                results = await self.cognee.search(
                    query_type=SearchType[query_type.upper()],
                    query_text=query_text,
                    top_k=top_k
                )
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
        mode : str
            Deletion mode ("soft" or "hard")

        Returns
        -------
        Dict[str, Any]
            Result of the deletion
        """
        if self.use_api:
            # API mode: Make HTTP request
            endpoint = f"{self.api_url}/api/v1/delete"
            params = {"data_id": str(data_id), "dataset_id": str(dataset_id), "mode": mode}

            response = await self.client.delete(
                endpoint, params=params, headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        else:
            # Direct mode: Call cognee directly
            from cognee.modules.users.methods import get_default_user

            with redirect_stdout(sys.stderr):
                user = await get_default_user()
                result = await self.cognee.delete(
                    data_id=data_id, dataset_id=dataset_id, mode=mode, user=user
                )
                return result

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

    async def get_pipeline_status(self, dataset_ids: List[UUID], pipeline_name: str) -> str:
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
        str
            Status information
        """
        if self.use_api:
            # Note: This would need a custom endpoint on the API side
            raise NotImplementedError("Pipeline status is not available via API")
        else:
            # Direct mode: Call cognee directly
            from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status

            with redirect_stdout(sys.stderr):
                status = await get_pipeline_status(dataset_ids, pipeline_name)
                return str(status)

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

    async def close(self):
        """Close the HTTP client if in API mode."""
        if self.use_api and hasattr(self, "client"):
            await self.client.aclose()
