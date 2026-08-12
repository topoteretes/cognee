"""Remote HTTP client that proxies V2 operations to a Cognee Cloud instance."""

import asyncio
import io
from pathlib import Path
from typing import Any, Optional, Union
from uuid import UUID

import aiohttp

from cognee.api.v1.serve.exceptions import (
    CogneeTransportError,
    http_error_for_status,
)
from cognee.api.v1.serve.state import UNSET, _Unset
from cognee.modules.ingestion.data_types.TextData import create_text_data
from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.cloud_client")


def _text_upload_filename(text: str) -> str:
    """Content-hash filename for raw-text uploads, via local ingestion's namer.

    Delegates to ``TextData`` — the same source ``save_data_to_file`` uses for
    nameless text (``text_<md5>.txt``). A fixed placeholder here instead makes
    every text upload for a tenant collide on one remote object, racing
    concurrent adds against the server's content-hash read-back
    (FileContentHashingError 409s).
    """
    return create_text_data(text).get_metadata()["name"]


def _as_node_set_list(node_set: Union[str, list, None]) -> Optional[list]:
    if node_set is None:
        return None
    if isinstance(node_set, str):
        return [node_set]
    return list(node_set)


class CloudClient:
    """Async HTTP client for a remote Cognee Cloud tenant instance.

    All requests use ``X-Api-Key`` for authentication, matching the
    SaaS backend's API key auth backend.

    Failures raise the typed errors from ``cognee.api.v1.serve.exceptions``
    (all subclasses of ``RuntimeError``): ``CogneeTransportError`` when the
    instance was never reached, ``CogneeAuthError`` / ``CogneeClientRequestError`` /
    ``CogneeServerError`` for HTTP 401·403 / other 4xx / 5xx responses.

    Every operation accepts ``timeout=<seconds>`` to bound that single call
    (integration hook paths run on read budgets of a few seconds); without
    it the class-level defaults apply.
    """

    def __init__(self, service_url: str, api_key: str):
        self.service_url = service_url.rstrip("/")
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    # Default for ordinary API calls: aiohttp's standard 5-minute total,
    # with connect failures surfacing quickly.
    # 600s: long-running blocking operations (e.g. cognify over a large
    # dataset) can legitimately take many minutes server-side
    DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=600, sock_connect=30)
    # Archive uploads (cognee.push) plus the synchronous server-side import
    # can legitimately exceed any fixed total; per-read inactivity stays
    # bounded instead. Applied per-request, only to archive uploads.
    UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-Api-Key": self.api_key},
                timeout=self.DEFAULT_TIMEOUT,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _health_check(self) -> bool:
        """Verify the remote instance is reachable."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.service_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    # ----- request plumbing -----

    def _resolve_timeout(
        self,
        timeout: Optional[float],
        default: Optional[aiohttp.ClientTimeout] = None,
    ) -> aiohttp.ClientTimeout:
        if timeout is None:
            return default or self.DEFAULT_TIMEOUT
        return aiohttp.ClientTimeout(total=timeout, sock_connect=min(30.0, timeout))

    async def _request(
        self,
        method: str,
        operation: str,
        path: str,
        *,
        json: Optional[dict] = None,
        data: Any = None,
        params: Any = None,
        timeout: Optional[float] = None,
        default_timeout: Optional[aiohttp.ClientTimeout] = None,
    ) -> Any:
        """Send a request to the instance, mapping failures to typed errors."""
        session = await self._get_session()
        request_timeout = self._resolve_timeout(timeout, default_timeout)
        request_kwargs: dict = {"timeout": request_timeout}
        if params is not None:
            request_kwargs["params"] = params
        if json is not None:
            request_kwargs["json"] = json
        if data is not None:
            request_kwargs["data"] = data
        send = getattr(session, method.lower())
        try:
            async with send(f"{self.service_url}{path}", **request_kwargs) as resp:
                if resp.status >= 400:
                    body: Any = await resp.text()
                    try:
                        import json as json_module

                        body = json_module.loads(body)
                    except Exception:
                        pass
                    raise http_error_for_status(resp.status, body, operation=operation)
                if resp.status == 204:
                    return None
                try:
                    return await resp.json()
                except Exception:
                    # Some routes (e.g. dataset deletion) reply with an empty
                    # or non-JSON body on success.
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise CogneeTransportError(
                f"Could not reach the Cognee instance at {self.service_url} "
                f"during {operation}: {error}",
                operation=operation,
                cause=error,
            ) from error

    async def _post(
        self,
        operation: str,
        path: str,
        *,
        json: Optional[dict] = None,
        data: Any = None,
        timeout: Optional[float] = None,
        default_timeout: Optional[aiohttp.ClientTimeout] = None,
    ) -> Any:
        return await self._request(
            "POST",
            operation,
            path,
            json=json,
            data=data,
            timeout=timeout,
            default_timeout=default_timeout,
        )

    # ----- V2 Operations -----

    async def remember(self, data: Any, dataset_name: str = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/remember — ingest data and build knowledge graph."""
        form = aiohttp.FormData()
        form.add_field("datasetName", dataset_name)

        if kwargs.get("dataset_id"):
            form.add_field("datasetId", str(kwargs["dataset_id"]))
        if kwargs.get("session_id"):
            form.add_field("session_id", kwargs["session_id"])
        if kwargs.get("run_in_background"):
            form.add_field("run_in_background", "true")
        if kwargs.get("custom_prompt"):
            form.add_field("custom_prompt", kwargs["custom_prompt"])
        if kwargs.get("chunk_size") is not None:
            form.add_field("chunk_size", str(kwargs["chunk_size"]))
        if kwargs.get("chunks_per_batch") is not None:
            form.add_field("chunks_per_batch", str(kwargs["chunks_per_batch"]))
        # node_set drives the integrations' categorization scheme
        # (user_context / project_docs / agent_actions / qa / trace);
        # sent as a repeated form field, same as the server expects.
        for node_set_entry in _as_node_set_list(kwargs.get("node_set")) or []:
            form.add_field("node_set", str(node_set_entry))
        content_type_kw = kwargs.get("content_type")
        if content_type_kw is not None:
            form.add_field("content_type", str(content_type_kw))
        if kwargs.get("import_mode") is not None:
            form.add_field("import_mode", str(kwargs["import_mode"]))

        # Skills are local SKILL.md files. The server's add_skills() reads
        # paths from its own filesystem — sending the path string verbatim
        # would have the server look for that path on the POD, not the
        # caller. For content_type="skills", read each SKILL.md and upload
        # its bytes so the server can write them to a tempdir.
        if content_type_kw == "skills" and isinstance(data, (str, Path)):
            source = Path(data).expanduser()
            if source.is_file():
                skill_files = [source] if source.name == "SKILL.md" else []
            elif source.is_dir():
                skill_files = sorted(source.rglob("SKILL.md"))
            else:
                raise FileNotFoundError(f"Skills source not found: {data}")
            if not skill_files:
                raise ValueError(f"No SKILL.md files under {data}")
            base = source if source.is_dir() else source.parent
            for skill_path in skill_files:
                # Preserve relative structure so the server can reconstruct
                # the SKILL.md layout when writing to its tempdir.
                rel = skill_path.relative_to(base).as_posix()
                form.add_field("data", skill_path.open("rb"), filename=rel)
        # Handle data — string or file-like objects
        elif isinstance(data, str):
            form.add_field(
                "data",
                io.BytesIO(data.encode("utf-8")),
                filename=_text_upload_filename(data),
                content_type="text/plain",
            )
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    form.add_field(
                        "data",
                        io.BytesIO(item.encode("utf-8")),
                        filename=_text_upload_filename(item),
                        content_type="text/plain",
                    )
                elif hasattr(item, "read"):
                    name = getattr(item, "name", "upload")
                    form.add_field("data", item, filename=name)
        elif hasattr(data, "read"):
            name = getattr(data, "name", "upload")
            form.add_field("data", data, filename=name)

        default_timeout = (
            self.UPLOAD_TIMEOUT
            if kwargs.get("content_type") == "cogx-archive"
            else self.DEFAULT_TIMEOUT
        )
        return await self._post(
            "remember",
            "/api/v1/remember",
            data=form,
            timeout=kwargs.get("timeout"),
            default_timeout=default_timeout,
        )

    async def remember_entry(
        self,
        entry,
        dataset_name: str = "main_dataset",
        session_id: Optional[str] = None,
        skill_improvement: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """POST /api/v1/remember/entry — store a typed MemoryEntry.

        ``entry`` is a pydantic MemoryEntry.
        """
        # Pydantic v2: model_dump preserves the discriminator field.
        entry_dump = entry.model_dump(mode="json")

        payload = {
            "entry": entry_dump,
            "dataset_name": dataset_name,
            "session_id": session_id,
            "skill_improvement": skill_improvement,
        }

        return await self._post(
            "remember_entry", "/api/v1/remember/entry", json=payload, timeout=timeout
        )

    async def recall(
        self,
        query_text: str,
        query_type: Any = UNSET,
        **kwargs,
    ) -> list:
        """POST /api/v1/recall — query the knowledge graph and/or session cache.

        ``query_type`` is tri-state:

        - left as ``UNSET`` — the key is omitted and the server applies its
          backward-compatible default (GRAPH_COMPLETION);
        - explicit ``None`` — sent as ``"search_type": null``, opting into
          server-side auto-routing and session-scope reads;
        - a value — sent as that search type.
        """
        payload: dict = {"query": query_text}
        if not isinstance(query_type, _Unset):
            if query_type is None:
                payload["search_type"] = None
            else:
                payload["search_type"] = (
                    query_type if isinstance(query_type, str) else query_type.value
                )
        if kwargs.get("dataset_ids"):
            payload["dataset_ids"] = [str(dataset_id) for dataset_id in kwargs["dataset_ids"]]
        elif kwargs.get("datasets"):
            payload["datasets"] = kwargs["datasets"]
        if kwargs.get("top_k"):
            payload["top_k"] = kwargs["top_k"]
        if kwargs.get("system_prompt"):
            payload["system_prompt"] = kwargs["system_prompt"]
        if kwargs.get("node_name"):
            payload["node_name"] = kwargs["node_name"]
        if kwargs.get("only_context"):
            payload["only_context"] = kwargs["only_context"]
        if kwargs.get("verbose"):
            payload["verbose"] = kwargs["verbose"]
        if kwargs.get("session_id"):
            payload["session_id"] = kwargs["session_id"]
        if kwargs.get("scope") is not None:
            payload["scope"] = kwargs["scope"]
        if kwargs.get("context_profile") is not None:
            payload["context_profile"] = kwargs["context_profile"]
        if kwargs.get("include_references") is not None:
            payload["include_references"] = kwargs["include_references"]
        if kwargs.get("response_schema") is not None:
            payload["response_schema"] = kwargs["response_schema"]
        if kwargs.get("tool_connections") is not None:
            payload["tool_connections"] = kwargs["tool_connections"]
        if kwargs.get("tools_trigger") not in (None, "always"):
            payload["tools_trigger"] = kwargs["tools_trigger"]

        return await self._post(
            "recall", "/api/v1/recall", json=payload, timeout=kwargs.get("timeout")
        )

    async def improve(self, dataset: Any = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/improve — enrich the knowledge graph.

        ``session_ids`` bridges session feedback and Q&A into the permanent
        graph — the server runs the full session pipeline when it is set.
        """
        payload: dict = {}
        if isinstance(dataset, UUID):
            payload["dataset_id"] = str(dataset)
        else:
            payload["dataset_name"] = str(dataset)
        if kwargs.get("run_in_background"):
            payload["run_in_background"] = True
        if kwargs.get("node_name"):
            payload["node_name"] = kwargs["node_name"]
        if kwargs.get("session_ids"):
            payload["session_ids"] = list(kwargs["session_ids"])
        if kwargs.get("extraction_tasks"):
            payload["extraction_tasks"] = kwargs["extraction_tasks"]
        if kwargs.get("enrichment_tasks"):
            payload["enrichment_tasks"] = kwargs["enrichment_tasks"]
        if kwargs.get("data"):
            payload["data"] = kwargs["data"]
        if kwargs.get("build_global_context_index"):
            payload["build_global_context_index"] = True

        return await self._post(
            "improve", "/api/v1/improve", json=payload, timeout=kwargs.get("timeout")
        )

    # ----- V1 Operations (add / cognify / search) -----

    async def add(self, data: Any, dataset_name: str = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/add — ingest data into a dataset."""
        form = aiohttp.FormData()
        form.add_field("datasetName", dataset_name)
        if kwargs.get("dataset_id"):
            form.add_field("datasetId", str(kwargs["dataset_id"]))
        if kwargs.get("run_in_background"):
            form.add_field("run_in_background", "true")
        for node_set_entry in _as_node_set_list(kwargs.get("node_set")) or []:
            form.add_field("node_set", str(node_set_entry))

        if isinstance(data, str):
            form.add_field(
                "data",
                io.BytesIO(data.encode("utf-8")),
                filename=_text_upload_filename(data),
                content_type="text/plain",
            )
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    form.add_field(
                        "data",
                        io.BytesIO(item.encode("utf-8")),
                        filename=_text_upload_filename(item),
                        content_type="text/plain",
                    )
                elif hasattr(item, "read"):
                    name = getattr(item, "name", "upload")
                    form.add_field("data", item, filename=name)
        elif hasattr(data, "read"):
            name = getattr(data, "name", "upload")
            form.add_field("data", data, filename=name)

        return await self._post("add", "/api/v1/add", data=form, timeout=kwargs.get("timeout"))

    async def cognify(self, datasets: Any = None, **kwargs) -> dict:
        """POST /api/v1/cognify — build the knowledge graph."""
        payload: dict = {}
        if datasets:
            payload["datasets"] = (
                [str(d) for d in datasets] if isinstance(datasets, list) else [str(datasets)]
            )
        if kwargs.get("run_in_background"):
            payload["run_in_background"] = True
        if kwargs.get("custom_prompt"):
            payload["custom_prompt"] = kwargs["custom_prompt"]
        if kwargs.get("chunk_size") is not None:
            payload["chunk_size"] = kwargs["chunk_size"]
        if kwargs.get("chunks_per_batch") is not None:
            payload["chunks_per_batch"] = kwargs["chunks_per_batch"]

        return await self._post(
            "cognify", "/api/v1/cognify", json=payload, timeout=kwargs.get("timeout")
        )

    async def search(self, query: str, **kwargs) -> list:
        """POST /api/v1/search — query the knowledge graph."""
        payload: dict = {"query": query}
        if kwargs.get("search_type"):
            st = kwargs["search_type"]
            payload["searchType"] = st if isinstance(st, str) else st.value
        if kwargs.get("datasets"):
            payload["datasets"] = kwargs["datasets"]
        if kwargs.get("dataset_ids"):
            dataset_ids = kwargs["dataset_ids"]
            if isinstance(dataset_ids, UUID):
                dataset_ids = [dataset_ids]
            payload["datasetIds"] = [str(dataset_id) for dataset_id in dataset_ids]
        if kwargs.get("top_k") is not None:
            payload["topK"] = kwargs["top_k"]
        if kwargs.get("system_prompt"):
            payload["systemPrompt"] = kwargs["system_prompt"]
        if kwargs.get("node_name"):
            payload["nodeName"] = kwargs["node_name"]
        if kwargs.get("only_context") is not None:
            payload["onlyContext"] = kwargs["only_context"]
        if kwargs.get("verbose") is not None:
            payload["verbose"] = kwargs["verbose"]
        if kwargs.get("skills") is not None:
            payload["skills"] = [
                skill.name if hasattr(skill, "name") else str(skill) for skill in kwargs["skills"]
            ]
        if kwargs.get("tools") is not None:
            payload["tools"] = kwargs["tools"]
        if kwargs.get("max_iter") is not None:
            payload["maxIter"] = kwargs["max_iter"]
        if kwargs.get("include_references") is not None:
            payload["includeReferences"] = kwargs["include_references"]
        if kwargs.get("code_query") is not None:
            payload["codeQuery"] = kwargs["code_query"]

        return await self._post(
            "search", "/api/v1/search", json=payload, timeout=kwargs.get("timeout")
        )

    async def forget(self, **kwargs) -> dict:
        """POST /api/v1/forget — delete data from the knowledge graph."""
        payload = {}
        if kwargs.get("everything"):
            payload["everything"] = True
        if kwargs.get("dataset"):
            payload["dataset"] = str(kwargs["dataset"])
        if kwargs.get("dataset_id"):
            payload["dataset_id"] = str(kwargs["dataset_id"])
        if kwargs.get("data_id"):
            payload["data_id"] = str(kwargs["data_id"])
        if kwargs.get("memory_only") is not None:
            payload["memory_only"] = bool(kwargs["memory_only"])

        return await self._post(
            "forget", "/api/v1/forget", json=payload, timeout=kwargs.get("timeout")
        )

    # ----- Datasets -----

    async def datasets_list(self, timeout: Optional[float] = None) -> list:
        """GET /api/v1/datasets — list the caller's datasets."""
        return await self._request("GET", "datasets_list", "/api/v1/datasets", timeout=timeout)

    async def datasets_create(self, name: str, timeout: Optional[float] = None) -> dict:
        """POST /api/v1/datasets — get-or-create a dataset by name.

        The server returns the existing dataset when the name is already
        taken, so this call is safe to repeat (the integrations' "ensure
        dataset" pattern).
        """
        return await self._request(
            "POST", "datasets_create", "/api/v1/datasets", json={"name": name}, timeout=timeout
        )

    async def datasets_status(
        self,
        dataset_ids: list,
        pipelines: Optional[list] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """GET /api/v1/datasets/status — pipeline status per dataset.

        The only way to track ``run_in_background`` work: poll with the
        pipeline names of interest (``cognify_pipeline``, ``memify_pipeline``).
        """
        params = [("dataset", str(dataset_id)) for dataset_id in dataset_ids]
        if pipelines:
            params.extend(("pipeline", pipeline) for pipeline in pipelines)
        return await self._request(
            "GET", "datasets_status", "/api/v1/datasets/status", params=params, timeout=timeout
        )

    async def datasets_data(self, dataset_id, timeout: Optional[float] = None) -> list:
        """GET /api/v1/datasets/{id}/data — list data items in a dataset."""
        return await self._request(
            "GET", "datasets_data", f"/api/v1/datasets/{dataset_id}/data", timeout=timeout
        )

    async def datasets_delete(self, dataset_id, timeout: Optional[float] = None) -> None:
        """DELETE /api/v1/datasets/{id} — delete one dataset."""
        return await self._request(
            "DELETE", "datasets_delete", f"/api/v1/datasets/{dataset_id}", timeout=timeout
        )

    async def datasets_delete_all(self, timeout: Optional[float] = None) -> None:
        """DELETE /api/v1/datasets — delete all of the caller's datasets."""
        return await self._request(
            "DELETE", "datasets_delete_all", "/api/v1/datasets", timeout=timeout
        )

    # ----- Agent connections -----

    async def agents_register(
        self,
        agent_session_name: str,
        *,
        type: str = "api",
        memory_mode: str = "unknown",
        session_id: Optional[str] = None,
        dataset_names: Optional[list] = None,
        dataset_ids: Optional[list] = None,
        source: str = "api",
        origin_function: Optional[str] = None,
        metadata: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """POST /api/v1/agents/register — announce an agent connection."""
        payload: dict = {
            "agent_session_name": agent_session_name,
            "type": type,
            "memory_mode": memory_mode,
            "source": source,
        }
        if session_id:
            payload["session_id"] = session_id
        if dataset_names:
            payload["dataset_names"] = list(dataset_names)
        if dataset_ids:
            payload["dataset_ids"] = [str(dataset_id) for dataset_id in dataset_ids]
        if origin_function:
            payload["origin_function"] = origin_function
        if metadata:
            payload["metadata"] = metadata
        return await self._post(
            "agents_register", "/api/v1/agents/register", json=payload, timeout=timeout
        )

    async def agents_unregister(
        self, agent_session_name: str, timeout: Optional[float] = None
    ) -> dict:
        """POST /api/v1/agents/unregister — deactivate an agent connection."""
        return await self._post(
            "agents_unregister",
            "/api/v1/agents/unregister",
            json={"agent_session_name": agent_session_name},
            timeout=timeout,
        )

    async def agents_connections_me(
        self,
        agent_session_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """GET /api/v1/agents/connections/me — the caller's connection state."""
        params = {"agent_session_name": agent_session_name} if agent_session_name else None
        return await self._request(
            "GET",
            "agents_connections_me",
            "/api/v1/agents/connections/me",
            params=params,
            timeout=timeout,
        )

    # ----- Users -----

    async def users_me(self, timeout: Optional[float] = None) -> dict:
        """GET /api/v1/users/me — resolve the authenticated principal."""
        return await self._request("GET", "users_me", "/api/v1/users/me", timeout=timeout)
