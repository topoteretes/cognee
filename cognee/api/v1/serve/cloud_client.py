"""Remote HTTP client that proxies V2 operations to a Cognee Cloud instance."""

import asyncio
import io
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import aiohttp

from cognee.modules.conflicts.models import ConflictDetail
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


def remote_graph_schema(graph_model: Any) -> dict | None:
    """The ``graph_model`` a remote remember or cognify sends: the model as the JSON
    schema the server rebuilds with ``graph_schema_to_graph_model``, typed ``Edge`` and
    ``FromIdentity`` fields included, or None for the server's default ``KnowledgeGraph``.

    A model the LLM extraction could not fill raises here, before anything is uploaded.
    """
    from cognee.shared.data_models import KnowledgeGraph
    from cognee.shared.graph_model_utils import graph_model_to_graph_schema

    if graph_model is None or graph_model is KnowledgeGraph:
        return None
    return graph_model_to_graph_schema(graph_model)


def _node_type_name(node_type: Any) -> str | None:
    """The ``node_type`` wire form: the class name, or None for the NodeSet default.

    Omitting the field is how the server is told "node-set tags", so the default
    travels as nothing and only a typed graph scope names its class.
    """
    from cognee.modules.engine.models.node_set import NodeSet

    if node_type is None or node_type is NodeSet:
        return None
    return node_type.__name__


def _failed_update_result(body: str) -> dict | None:
    """Parse a 500 body as an update result when it is one with status "failed"."""
    from cognee.api.v1.update.result import UpdateResult

    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if isinstance(payload, dict) and payload.get("status") == "failed" and "data_id" in payload:
        return UpdateResult.model_validate(payload).model_dump()
    return None


class CloudClient:
    """Async HTTP client for a remote Cognee Cloud tenant instance.

    All requests use ``X-Api-Key`` for authentication, matching the
    SaaS backend's API key auth backend.
    """

    def __init__(self, service_url: str, api_key: str):
        self.service_url = service_url.rstrip("/")
        self.api_key = api_key
        # Tenant /api/v1/users/me reported at connect time; read by telemetry
        # while this client is the active connection.
        self.tenant_id = ""
        self._session: aiohttp.ClientSession | None = None

    # Default for ordinary API calls: aiohttp's standard 5-minute total,
    # with connect failures surfacing quickly.
    # 600s: long-running blocking operations (e.g. cognify over a large
    # dataset) can legitimately take many minutes server-side
    DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=600, sock_connect=30)
    # Archive uploads (cognee.push) plus the synchronous server-side import
    # can legitimately exceed any fixed total; per-read inactivity stays
    # bounded instead. Applied per-request, only to archive uploads.
    UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
    # The tenant lookup only feeds telemetry; a slow instance must not delay
    # serve(), so it gives up quickly and the connection proceeds without it.
    TENANT_LOOKUP_TIMEOUT = aiohttp.ClientTimeout(total=5)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # trust_env: honour HTTP(S)_PROXY / NO_PROXY like every other HTTP
            # client cognee uses. aiohttp ignores them by default, which in a
            # proxied environment (corporate egress, Docker Sandboxes' credential
            # proxy) sends requests around the proxy — a proxy-managed API key
            # then never gets substituted and the server sees the placeholder.
            self._session = aiohttp.ClientSession(
                headers={"X-Api-Key": self.api_key},
                timeout=self.DEFAULT_TIMEOUT,
                trust_env=True,
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
            logger.debug(
                "Falling back to False after error in CloudClient._health_check", exc_info=True
            )
            return False

    async def _auth_check(self) -> int | None:
        """Status of an authenticated probe, or None when unreachable.

        ``/health`` is unauthenticated, so it cannot tell a working API key
        from a rejected one. Probing an authenticated endpoint lets serve()
        fail at connect time instead of on the first real operation.
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.service_url}/api/v1/datasets") as resp:
                return resp.status
        except Exception:
            logger.debug(
                "Falling back to None after error in CloudClient._auth_check", exc_info=True
            )
            return None

    async def _get_tenant_id(self) -> str:
        """Tenant id of the connected API key's user, or "" when there is none.

        Every instance answers ``/api/v1/users/me`` with fastapi-users'
        ``UserRead``. On a Cloud tenant instance its ``tenant_id`` is the
        control-plane ``tenants.id``; a self-hosted server reports its own
        local tenant. Both are recorded — analytics tells them apart by host.
        Any failure, including a timeout after ``TENANT_LOOKUP_TIMEOUT``,
        returns "": this must never fail or stall a connection.
        """
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.service_url}/api/v1/users/me", timeout=self.TENANT_LOOKUP_TIMEOUT
            ) as resp:
                if resp.status != 200:
                    return ""
                data = await resp.json()
            tenant_id = data.get("tenant_id") if isinstance(data, dict) else None
            return str(UUID(str(tenant_id))) if tenant_id else ""
        except Exception:
            logger.debug(
                "Falling back to no tenant after error in CloudClient._get_tenant_id",
                exc_info=True,
            )
            return ""

    # ----- V2 Operations -----

    async def remember(self, data: Any, dataset_name: str = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/remember — ingest data and build knowledge graph."""
        # Resolved first: a model the route cannot carry raises before any upload.
        graph_schema = remote_graph_schema(kwargs.get("graph_model"))
        session = await self._get_session()

        form = aiohttp.FormData()
        form.add_field("datasetName", dataset_name)
        if graph_schema is not None:
            form.add_field("graph_model", json.dumps(graph_schema))

        if kwargs.get("dataset_id"):
            form.add_field("datasetId", str(kwargs["dataset_id"]))
        if kwargs.get("session_id"):
            form.add_field("session_id", kwargs["session_id"])
        if kwargs.get("run_in_background"):
            form.add_field("run_in_background", "true")
        if kwargs.get("self_improvement") is not None:
            form.add_field("self_improvement", "true" if kwargs["self_improvement"] else "false")
        if kwargs.get("custom_prompt"):
            form.add_field("custom_prompt", kwargs["custom_prompt"])
        if kwargs.get("chunk_size") is not None:
            form.add_field("chunk_size", str(kwargs["chunk_size"]))
        if kwargs.get("chunks_per_batch") is not None:
            form.add_field("chunks_per_batch", str(kwargs["chunks_per_batch"]))
        content_type_kw = kwargs.get("content_type")
        if content_type_kw is not None:
            form.add_field("content_type", str(content_type_kw))
        if kwargs.get("import_mode") is not None:
            form.add_field("import_mode", str(kwargs["import_mode"]))

        # Code repos travel as spec strings in the 'raw_data' form field —
        # the server clones git URLs itself and reads local paths from its own
        # filesystem (only useful when it shares the caller's filesystem).
        # Nothing is uploaded.
        if content_type_kw == "code":
            specs = data if isinstance(data, list) else [data]
            for spec in specs:
                form.add_field("raw_data", str(spec))
            if kwargs.get("index_vectors"):
                form.add_field("index_vectors", "true")
        # Skills are local SKILL.md files. The server's add_skills() reads
        # paths from its own filesystem — sending the path string verbatim
        # would have the server look for that path on the POD, not the
        # caller. For content_type="skills", read each SKILL.md and upload
        # its bytes so the server can write them to a tempdir.
        elif content_type_kw == "skills" and isinstance(data, (str, Path)):
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

        # Code ingestion can block on a clone + whole-repo parse; the archive
        # timeout (no total cap) fits both. Prefer run_in_background=True for
        # large repos regardless.
        timeout = (
            self.UPLOAD_TIMEOUT
            if kwargs.get("content_type") in ("cogx-archive", "code")
            else self.DEFAULT_TIMEOUT
        )
        async with session.post(
            f"{self.service_url}/api/v1/remember", data=form, timeout=timeout
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote remember failed ({resp.status}): {body}")
            return await resp.json()

    async def remember_entry(
        self,
        entry,
        dataset_name: str = "main_dataset",
        session_id: str | None = None,
        skill_improvement: dict | None = None,
        dataset_id: str | None = None,
    ) -> dict:
        """POST /api/v1/remember/entry — store a typed MemoryEntry.

        ``entry`` is a pydantic MemoryEntry.
        """
        session = await self._get_session()

        # Pydantic v2: model_dump preserves the discriminator field.
        entry_dump = entry.model_dump(mode="json")

        payload = {
            "entry": entry_dump,
            "dataset_name": dataset_name,
            "session_id": session_id,
            "skill_improvement": skill_improvement,
        }

        if dataset_id is not None:
            payload["dataset_id"] = str(dataset_id)

        async with session.post(
            f"{self.service_url}/api/v1/remember/entry",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote remember_entry failed ({resp.status}): {body}")
            return await resp.json()

    async def recall(self, query_text: str, query_type: str | None = None, **kwargs) -> list:
        """POST /api/v1/recall — query the knowledge graph and/or session cache."""
        session = await self._get_session()

        payload: dict = {"query": query_text}
        if query_type:
            payload["search_type"] = query_type if isinstance(query_type, str) else query_type.value
        if kwargs.get("dataset_ids"):
            payload["dataset_ids"] = [str(dataset_id) for dataset_id in kwargs["dataset_ids"]]
        elif kwargs.get("datasets"):
            payload["datasets"] = kwargs["datasets"]
        if kwargs.get("top_k"):
            payload["top_k"] = kwargs["top_k"]
        if kwargs.get("system_prompt"):
            payload["system_prompt"] = kwargs["system_prompt"]
        node_type_name = await self._node_type_payload(
            kwargs.get("node_type"), kwargs.get("node_name")
        )
        if node_type_name is not None:
            payload["node_type"] = node_type_name
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
        if kwargs.get("code_query") is not None:
            payload["code_query"] = kwargs["code_query"]

        async with session.post(
            f"{self.service_url}/api/v1/recall",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote recall failed ({resp.status}): {body}")
            return await resp.json()

    async def improve(self, dataset: Any = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/improve — enrich the knowledge graph."""
        session = await self._get_session()

        payload = {}
        if isinstance(dataset, UUID):
            payload["dataset_id"] = str(dataset)
        else:
            payload["dataset_name"] = str(dataset)
        if kwargs.get("run_in_background"):
            payload["run_in_background"] = True
        if kwargs.get("session_ids"):
            payload["session_ids"] = list(kwargs["session_ids"])
        if kwargs.get("build_global_context_index"):
            payload["build_global_context_index"] = True
        if kwargs.get("build_truth_subspace"):
            payload["build_truth_subspace"] = True
        if kwargs.get("review_conflicts") is not None:
            payload["review_conflicts"] = kwargs["review_conflicts"]
        if kwargs.get("feedback_alpha") is not None:
            payload["feedback_alpha"] = kwargs["feedback_alpha"]
        # Deprecated custom tasks (SDK-1115): the /improve DTO takes registry task
        # names, a data string and node_name. Task objects and the other scoping
        # options cannot cross the wire, so they fail loudly instead of the
        # server silently running without them.
        for key in ("extraction_tasks", "enrichment_tasks"):
            tasks = kwargs.get(key)
            if tasks:
                if not all(isinstance(task, str) for task in tasks):
                    raise ValueError(
                        f"improve({key}=...) on a remote instance takes registry "
                        "task names (strings); Task objects cannot be serialized."
                    )
                payload[key] = list(tasks)
        if kwargs.get("data") is not None:
            payload["data"] = kwargs["data"]
        if kwargs.get("node_name") is not None:
            payload["node_name"] = kwargs["node_name"]
        for key in ("node_type", "vector_db_config", "graph_db_config"):
            if kwargs.get(key) is not None:
                raise ValueError(
                    f"improve({key}=...) is not supported on a remote instance; run it locally."
                )

        async with session.post(
            f"{self.service_url}/api/v1/improve",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote improve failed ({resp.status}): {body}")
            return await resp.json()

    # ----- V1 Operations (add / cognify / search) -----

    async def add(self, data: Any, dataset_name: str = "main_dataset", **kwargs) -> dict:
        """POST /api/v1/add — ingest data into a dataset."""
        session = await self._get_session()

        form = aiohttp.FormData()
        form.add_field("datasetName", dataset_name)

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

        async with session.post(f"{self.service_url}/api/v1/add", data=form) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote add failed ({resp.status}): {body}")
            return await resp.json()

    async def update(
        self,
        data_id: UUID,
        data: Any,
        dataset_id: UUID,
        node_set: list | None = None,
        chunk_level_diff: bool = True,
    ) -> dict:
        """PATCH /api/v1/update — replace one document in place on the remote.

        Mirrors the route: ``data_id``, ``dataset_id`` and ``chunk_level_diff``
        travel as query params, the new content as the multipart ``data`` file,
        ``node_set`` as repeated form fields. The server keeps the document's
        id across the update, so this is a real replace. ``data=None`` sends no
        file: the server then replaces only the node sets. ``node_set=None``
        sends no field (keep) and ``[]`` the route's clear value.
        """
        from cognee.api.v1.update.routers.get_update_router import CLEAR_NODE_SET

        if data is None and node_set is None:
            raise ValueError("update() needs replacement data, a node_set, or both.")
        # update() replaces exactly one document; the local implementation
        # unwraps single-item lists and DataItem wrappers the same way.
        if isinstance(data, list):
            if len(data) != 1:
                raise ValueError(f"update() replaces exactly one document; got {len(data)} items.")
            data = data[0]
        if hasattr(data, "data") and hasattr(data, "data_id") and not hasattr(data, "read"):
            data = data.data

        session = await self._get_session()

        # Always multipart, as the route's file field expects, even without a file.
        form = aiohttp.FormData(default_to_multipart=True)
        if data is None:
            pass
        elif isinstance(data, str):
            form.add_field(
                "data",
                io.BytesIO(data.encode("utf-8")),
                filename=_text_upload_filename(data),
                content_type="text/plain",
            )
        elif hasattr(data, "read"):
            name = getattr(data, "name", "upload")
            form.add_field("data", data, filename=Path(name).name or "upload")
        else:
            raise TypeError(
                f"update() over serve() accepts text or a file object; got {type(data)}"
            )
        if node_set is not None:
            tags = [str(tag) for tag in node_set if tag]
            for tag in tags or [CLEAR_NODE_SET]:
                form.add_field("node_set", tag)

        params = {
            "data_id": str(data_id),
            "dataset_id": str(dataset_id),
            "chunk_level_diff": "true" if chunk_level_diff else "false",
        }
        from cognee.api.v1.update.result import UpdateResult

        async with session.patch(
            f"{self.service_url}/api/v1/update", params=params, data=form
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                # A failed rebuild travels with a 500 but is still a result, in
                # the same shape the local path returns, so the caller can read
                # the error and retry. Anything else is a remote error.
                failed = _failed_update_result(body)
                if failed is not None:
                    return failed
                raise RuntimeError(f"Remote update failed ({resp.status}): {body}")
            # Through the schema so the dict matches the local result exactly:
            # UUIDs as UUID objects, the fallback reason as its enum member.
            return UpdateResult.model_validate(await resp.json()).model_dump()

    # Feature probes. The connect-time checks (/health, an authenticated GET) say
    # the server is up and the key works, not which fields its routes know, so a
    # feature a server predates is detected from its OpenAPI schema. The schema is
    # fetched once per client; each probe caches its own answer. Class defaults so
    # instances built without ``__init__`` (tests) start unknown too.
    _openapi: dict | None = None
    _content_hash_lookup_supported: bool | None = None
    _node_type_supported: bool | None = None

    # The schema is a few hundred KiB; its own timeout keeps a slow docs route
    # from holding a probe under the 600 s request timeout.
    SCHEMA_PROBE_TIMEOUT = aiohttp.ClientTimeout(total=30)

    _CONTENT_HASH_UNSUPPORTED = (
        "Remote server does not support content_hash lookup on "
        "GET /api/v1/datasets/{dataset_id}/data: it predates the content_hash filter and "
        "sends no contentHash on its rows, so find_data() cannot tell a match from the "
        "rest of the dataset. Upgrade the server to use find_data() over serve()."
    )

    _NODE_TYPE_UNSUPPORTED = (
        "Remote server does not take node_type on POST /api/v1/search and "
        "POST /api/v1/recall: it predates the field and would read node_name as node-set "
        "tags, returning results for the wrong scope without an error. Upgrade the server "
        "to use a typed node_type over serve(); node_name alone (node-set tags) still works."
    )

    _NODE_TYPE_UNKNOWN = (
        "Could not read the remote server's schema (GET /openapi.json), so whether it "
        "takes node_type on POST /api/v1/search and POST /api/v1/recall is unknown. A "
        "server that predates the field would read node_name as node-set tags and return "
        "results for the wrong scope without an error, so the typed scope is not sent. "
        "Make /openapi.json reachable from this client; node_name alone (node-set tags) "
        "still works."
    )

    async def _server_openapi(self) -> dict | None:
        """The server's OpenAPI document, fetched once per client.

        ``None`` means it could not be read (no schema served, not JSON, or the
        read ran past ``SCHEMA_PROBE_TIMEOUT``) and is not cached, so a later call
        can still find out.
        """
        if self._openapi is not None:
            return self._openapi
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.service_url}/openapi.json", timeout=self.SCHEMA_PROBE_TIMEOUT
            ) as resp:
                if resp.status >= 400:
                    return None
                schema = await resp.json()
        # A total-timeout is a bare asyncio.TimeoutError (the builtin on 3.11+), not
        # a ClientError: only a connect-phase timeout is one.
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError):
            logger.warning("could not read the server schema from /openapi.json")
            return None
        if not isinstance(schema, dict):
            return None
        self._openapi = schema
        return schema

    async def supports_content_hash_lookup(self) -> bool | None:
        """Whether the server's dataset listing takes a ``content_hash`` query param.

        FastAPI drops an unknown query param without an error, so a server
        older than the content-hash lookup answers ``?content_hash=`` with the
        whole dataset listing, which ``find_data()`` would then hand back as
        matches. ``None`` means unknown (no schema served, or no listing route in
        it) and is not cached; the row-level check in ``list_data`` fails closed
        either way, so an unknown server is still asked.
        """
        if self._content_hash_lookup_supported is not None:
            return self._content_hash_lookup_supported
        schema = await self._server_openapi()
        if schema is None:
            return None
        try:
            parameters = schema["paths"]["/api/v1/datasets/{dataset_id}/data"]["get"].get(
                "parameters", []
            )
        except (KeyError, TypeError, AttributeError):
            logger.warning(
                "find_data: could not read the server schema to check content_hash support"
            )
            return None
        self._content_hash_lookup_supported = any(
            isinstance(parameter, dict)
            and parameter.get("name") == "content_hash"
            and parameter.get("in") == "query"
            for parameter in parameters
        )
        if not self._content_hash_lookup_supported:
            logger.warning(
                "find_data: the remote server predates the content_hash lookup; "
                "lookups over serve() are refused until it is upgraded"
            )
        return self._content_hash_lookup_supported

    async def supports_node_type(self) -> bool | None:
        """Whether the server's search and recall request bodies take ``nodeType``.

        Request bodies ignore unknown fields, so a server older than the field
        accepts the body and reads ``nodeName`` as node-set tags: no error, and
        results for the wrong scope. ``None`` means unknown (no schema served) and
        is not cached, so a later call can still find out.
        """
        if self._node_type_supported is not None:
            return self._node_type_supported
        schema = await self._server_openapi()
        if schema is None:
            return None
        try:
            schemas = schema["components"]["schemas"]
            self._node_type_supported = all(
                "nodeType" in schemas.get(name, {}).get("properties", {})
                for name in ("SearchPayloadDTO", "RecallPayloadDTO")
            )
        except (KeyError, TypeError, AttributeError):
            logger.warning("could not read the server schema to check node_type support")
            return None
        if not self._node_type_supported:
            logger.warning(
                "the remote server predates node_type on search and recall; "
                "typed scopes over serve() are refused until it is upgraded"
            )
        return self._node_type_supported

    async def _node_type_payload(self, node_type: Any, node_name: Any) -> str | None:
        """The ``node_type`` to send: None for the node-set default, else the class name.

        ``node_type`` only changes what ``node_name`` means, so without names the
        field has no effect and goes to any server unprobed. With names a typed
        scope is sent only to a server known to read it: an old server would apply
        the wrong scope, and a response carries nothing that would show it, so a
        server whose schema cannot be read is refused too. Dropping the type or the
        names instead would return results the caller did not ask for, with nothing
        to tell them.
        """
        node_type_name = _node_type_name(node_type)
        if node_type_name is None or not node_name:
            return node_type_name
        supported = await self.supports_node_type()
        if supported is None:
            raise RuntimeError(self._NODE_TYPE_UNKNOWN)
        if not supported:
            raise RuntimeError(self._NODE_TYPE_UNSUPPORTED)
        return node_type_name

    async def list_data(self, dataset_id: UUID, *, content_hash: str | None = None) -> list:
        """Traverse bounded pages, deduplicating overlaps during concurrent ingest.

        This is not a snapshot: inserts/deletes can make rows unreachable during
        offset traversal. Above offset 1,000,000 the server rejects requests;
        /data/count still reports the full dataset size.

        ``content_hash`` narrows the listing to the caller's rows holding that
        content. A server that knows the filter applies it (``?content_hash=``)
        and pages the matches; the rows are filtered again here on
        ``contentHash`` so a server that ignored the param can never turn a
        lookup into the whole dataset. A server known (from the schema probe)
        or found (rows without ``contentHash``) to predate the filter is
        refused with ``RuntimeError`` rather than guessed at: it has nothing
        the rows could be matched on.
        """
        if content_hash is not None and await self.supports_content_hash_lookup() is False:
            raise RuntimeError(self._CONTENT_HASH_UNSUPPORTED)

        session = await self._get_session()
        rows = []
        seen_ids = set()
        limit = 1000
        offset = 0
        while True:
            params: dict = {"limit": limit, "offset": offset}
            if content_hash is not None:
                params["content_hash"] = content_hash
            async with session.get(
                f"{self.service_url}/api/v1/datasets/{dataset_id}/data",
                params=params,
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"Remote list_data failed ({resp.status}): {body}")
                page = await resp.json()
            if not isinstance(page, list):
                raise RuntimeError("Remote list_data returned an invalid document list")
            page_ids = {str(row["id"]) for row in page if isinstance(row, dict) and "id" in row}
            if len(page) > limit or (page and page_ids and page_ids <= seen_ids):
                raise RuntimeError("Remote server did not honor dataset pagination")
            # Advance by consumed server rows, independently of deduplication.
            offset += len(page)
            for row in page:
                row_id = str(row["id"])
                if row_id not in seen_ids:
                    seen_ids.add(row_id)
                    rows.append(row)
            if len(page) < limit:
                break

        if content_hash is None:
            return rows
        return self._rows_holding(rows, content_hash)

    @classmethod
    def _rows_holding(cls, rows: list, content_hash: str) -> list:
        """Keep the rows whose ``contentHash`` is ``content_hash``; fail closed.

        A row without the key comes from a server that predates the lookup
        (its DataDTO had no ``contentHash``), so nothing in the listing can be
        matched and returning it unfiltered would hand the caller the whole
        dataset as "matches".
        """
        matches = []
        for row in rows:
            if not isinstance(row, dict) or "contentHash" not in row:
                raise RuntimeError(cls._CONTENT_HASH_UNSUPPORTED)
            if row["contentHash"] == content_hash:
                matches.append(row)
        return matches

    async def sparql(
        self,
        dataset_id: UUID,
        query: str,
        *,
        base_iri: str | None = None,
        limit: int = 1000,
    ) -> dict:
        """POST /api/v1/datasets/{dataset_id}/sparql — read-only SPARQL over the graph."""
        session = await self._get_session()

        payload: dict = {"query": query, "limit": limit}
        if base_iri:
            payload["baseIri"] = base_iri

        async with session.post(
            f"{self.service_url}/api/v1/datasets/{dataset_id}/sparql",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote SPARQL query failed ({resp.status}): {body}")
            return await resp.json()

    async def cognify(self, datasets: Any = None, **kwargs) -> dict:
        """POST /api/v1/cognify — build the knowledge graph."""
        graph_schema = remote_graph_schema(kwargs.get("graph_model"))
        session = await self._get_session()

        payload: dict = {}
        if graph_schema is not None:
            payload["graph_model"] = graph_schema
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

        async with session.post(
            f"{self.service_url}/api/v1/cognify",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote cognify failed ({resp.status}): {body}")
            return await resp.json()

    async def search(self, query: str, **kwargs) -> list:
        """POST /api/v1/search — query the knowledge graph."""
        session = await self._get_session()

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
        node_type_name = await self._node_type_payload(
            kwargs.get("node_type"), kwargs.get("node_name")
        )
        if node_type_name is not None:
            payload["nodeType"] = node_type_name
        if kwargs.get("node_name"):
            payload["nodeName"] = kwargs["node_name"]
        if kwargs.get("only_context") is not None:
            payload["onlyContext"] = kwargs["only_context"]
        if kwargs.get("session_id"):
            # Selects the session layer server-side: history and guidance for a real
            # completion, and the same layer inside an only_context string.
            payload["sessionId"] = kwargs["session_id"]
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

        async with session.post(
            f"{self.service_url}/api/v1/search",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote search failed ({resp.status}): {body}")
            return await resp.json()

    async def forget(self, **kwargs) -> dict:
        """POST /api/v1/forget — delete data from the knowledge graph."""
        session = await self._get_session()

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
        if kwargs.get("clean_history"):
            payload["clean_history"] = True

        async with session.post(
            f"{self.service_url}/api/v1/forget",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote forget failed ({resp.status}): {body}")
            return await resp.json()

    async def list_conflicts(
        self, dataset_id: UUID, *, status: str | None = None
    ) -> list[ConflictDetail]:
        """GET /api/v1/conflicts — a dataset's conflicts with their competing facts."""
        session = await self._get_session()

        params = {"dataset_id": str(dataset_id)}
        if status:
            params["status"] = status

        async with session.get(
            f"{self.service_url}/api/v1/conflicts",
            params=params,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote list_conflicts failed ({resp.status}): {body}")
            payload = await resp.json()
        if not isinstance(payload, list):
            raise RuntimeError("Remote list_conflicts returned an invalid conflict list")
        return [ConflictDetail.model_validate(item) for item in payload]

    async def resolve_conflict(
        self,
        conflict_id: UUID,
        *,
        dataset_id: UUID,
        selected_fact_id: UUID,
        reason: str | None = None,
    ) -> ConflictDetail:
        """POST /api/v1/conflicts/{conflict_id}/resolve — accept one competing fact."""
        session = await self._get_session()

        payload: dict = {"datasetId": str(dataset_id), "selectedFactId": str(selected_fact_id)}
        if reason is not None:
            payload["reason"] = reason

        async with session.post(
            f"{self.service_url}/api/v1/conflicts/{conflict_id}/resolve",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote resolve_conflict failed ({resp.status}): {body}")
            return ConflictDetail.model_validate(await resp.json())
