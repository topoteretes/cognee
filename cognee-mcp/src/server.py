import argparse
import asyncio
import base64
import importlib.metadata
import importlib.util
import os
import sys
from collections import deque
from contextlib import redirect_stdout
from datetime import datetime, timezone

import fastmcp
import uvicorn
from fastmcp import FastMCP
from fastmcp.server.transforms.search import BM25SearchTransform
from fastmcp.server.transforms.search.base import BaseSearchTransform
from fastmcp.server.http import HostOriginGuardMiddleware
from mcp import types
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.logging_utils import get_log_file_location, get_logger, setup_logging

try:
    from .cognee_client import CogneeClient
except ImportError:
    from cognee_client import CogneeClient

try:
    from .tool_registry import DEFAULT_TAG, MEMORY_TAG, ToolRegistry
except ImportError:
    from tool_registry import DEFAULT_TAG, MEMORY_TAG, ToolRegistry

try:
    from .server_utils import format_recall_results, parse_csv_list, validate_top_k
except ImportError:
    from server_utils import format_recall_results, parse_csv_list, validate_top_k


try:
    __version__ = importlib.metadata.version("cognee-mcp")
except importlib.metadata.PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0+unknown"

# Without an explicit version FastMCP reports *its own* package version in
# serverInfo, so every client showed "Cognee v3.4.6" (the FastMCP version)
# and there was no way to tell which cognee-mcp build was running.
mcp = FastMCP("Cognee", version=__version__)

# Tools register through this rather than @mcp.tool directly, so each one
# declares its tier at the definition site (see apply_tool_mode()).
registry = ToolRegistry(mcp)

logger = get_logger()

cognee_client: CogneeClient | None = None

# Per-dataset error ring buffer (bounded so long-running servers don't accumulate
# unbounded memory). Each entry is (iso_timestamp, error_message).
_TASK_ERROR_HISTORY = 50
_task_errors: dict[str, deque[tuple[str, str]]] = {}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Strong references to in-flight background tasks. asyncio's event loop only keeps
# weak references to tasks, so a fire-and-forget task can be GC'd mid-execution if
# the only reference is a local that went out of scope. Adding here pins them; the
# done_callback removes them on completion. See:
# https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
_background_tasks: set[asyncio.Task] = set()


def _track_background(coro) -> asyncio.Task:
    """Spawn a background task and pin it so the event loop won't GC it."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _record_task_error(dataset: str, error: str) -> None:
    """Append a background task error, bounded per-dataset."""
    bucket = _task_errors.setdefault(dataset, deque(maxlen=_TASK_ERROR_HISTORY))
    bucket.append((datetime.now(timezone.utc).isoformat(), error))


def _transport_security_kwargs(host: str) -> dict:
    """Build the Host/Origin guard kwargs for mcp.http_app() from env and bind host.

    FastMCP 3 takes these per-app rather than off a mutable settings object, so
    this returns kwargs instead of configuring global state.

    Env vars:
        MCP_DISABLE_DNS_REBINDING_PROTECTION: Set to "true" to disable all
            Host/Origin header validation. Useful for LAN or Docker deployments.
        MCP_ALLOWED_HOSTS: Comma-separated additional Host header patterns
            (e.g. "192.168.1.50:*,myserver.local:*"). Appended to the
            localhost defaults. Requires the ":*" port glob suffix.
    """
    disable = os.getenv("MCP_DISABLE_DNS_REBINDING_PROTECTION", "false").lower() == "true"

    if disable:
        logger.info("MCP transport security: DNS rebinding protection disabled")
        return {"host_origin_protection": False}

    extra_hosts = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]

    # "auto" only guards loopback binds and explicit allowlists. When the user
    # binds to 0.0.0.0 or a LAN IP, we must provide the full allowed list
    # ourselves and turn the guard on unconditionally.
    localhost_hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    localhost_origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]

    allowed_hosts = localhost_hosts + extra_hosts
    # Derive origins from extra hosts so users don't need to set both.
    allowed_origins = localhost_origins + [f"http://{h}" for h in extra_hosts]

    if host not in ("127.0.0.1", "localhost", "::1") or extra_hosts:
        logger.info(
            "MCP transport security: allowed_hosts=%s",
            allowed_hosts,
        )
        return {
            "host_origin_protection": True,
            "allowed_hosts": allowed_hosts,
            "allowed_origins": allowed_origins,
        }

    # Loopback-only with no extra hosts. Ask for "auto" explicitly rather than
    # falling through to FastMCP's own default, which is False — i.e. no guard
    # at all. DNS rebinding is precisely an attack on loopback services, so the
    # default bind is the one case that must not be left unguarded.
    logger.info("MCP transport security: Host/Origin guard in auto mode (loopback bind)")
    return {"host_origin_protection": "auto"}


TOOL_MODES = ("default", "minimal", "all")

# How many tools search_tools returns. Chosen from the recall sweep in
# tests/test_tool_search_benchmark.py: over a 500-tool catalog, BM25 recall@k
# goes 60% -> 80% between k=5 and k=10 and then plateaus, because the ranker was
# placing the right tool just outside a 5-wide window. Recall is the metric that
# matters — the agent sees every returned schema and picks for itself, so a tool
# missing from the window is unrecoverable while its rank inside the window
# barely costs anything.
#
# With today's small catalog this exceeds the unpinned count, so the window is
# never the binding constraint — but that does NOT mean every search returns
# every tool. BM25 drops zero-scoring tools, and its tokenizer does no
# stemming, so a query only matches tokens it literally contains. Misses come
# from vocabulary, not from k. When adding a tool, put the words an agent would
# actually use — in both singular and plural — in its description.
#
# The window costs context only on turns that call search, never on the
# per-turn tools/list payload.
TOOL_SEARCH_MAX_RESULTS = 10


def apply_tool_mode(mode: str | None = None) -> str:
    """Gate the advertised tool surface behind FastMCP's tool-search transform.

    Every tool stays registered and directly callable by name; the transform
    only changes what ``tools/list`` advertises, replacing the non-pinned tools
    with ``search_tools``/``call_tool``, so a fresh agent sees a handful of
    tools instead of the whole catalog.

    Modes (COGNEE_MCP_TOOL_MODE):
        default: pin the DEFAULT_TAG tools (memory API).
        minimal: pin only the memory API.
        all:     no transform, advertise everything (pre-3.x behavior).

    Returns the mode actually applied.
    """
    mode = (mode or os.getenv("COGNEE_MCP_TOOL_MODE") or "default").lower()

    if mode not in TOOL_MODES:
        logger.warning(
            "Unknown COGNEE_MCP_TOOL_MODE=%r; falling back to 'default'. Valid: %s",
            mode,
            ", ".join(TOOL_MODES),
        )
        mode = "default"

    # Drop any transform a previous call installed: add_transform() appends, so
    # calling this twice would otherwise stack search transforms on top of each
    # other and hide the first one's pinned tools.
    mcp._transforms = [t for t in mcp._transforms if not isinstance(t, BaseSearchTransform)]

    if mode == "all":
        logger.info("MCP tool mode 'all': advertising all %d tools", len(registry.tags))
        return mode

    pinned = registry.names_with_tag(DEFAULT_TAG if mode == "default" else MEMORY_TAG)
    mcp.add_transform(
        BM25SearchTransform(max_results=TOOL_SEARCH_MAX_RESULTS, always_visible=pinned)
    )
    logger.info(
        "MCP tool mode %r: advertising %s + search_tools/call_tool (%d tools searchable)",
        mode,
        pinned,
        len(registry.tags) - len(pinned),
    )
    return mode


def _is_running_in_docker() -> bool:
    """Check if the process is running inside a Docker container."""
    return os.path.exists("/.dockerenv") or os.path.isdir("/app")


def _get_cors_origins() -> list[str]:
    """Parse CORS allowed origins from MCP_CORS_ALLOW_ORIGINS env var."""
    raw = os.getenv("MCP_CORS_ALLOW_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _build_http_app(transport: str, host: str, path: str | None = None):
    """Build the ASGI app for an HTTP-family transport, guard and CORS included.

    Split out from _serve_with_cors so the transport security wiring can be
    asserted in-process, without binding a socket.

    `path` is forwarded to http_app() so --path actually moves the endpoint;
    without it the app always mounted at the FastMCP default while the startup
    banner advertised the requested path, so the logged URL 404'd.
    """
    security_kwargs = _transport_security_kwargs(host)
    extra_middleware: list[Middleware] = []

    # FastMCP installs its Host/Origin (DNS-rebinding) guard only on the
    # streamable-http app. create_sse_app() takes no such option in any released
    # version, so http_app() accepts these kwargs for transport="sse" and drops
    # them — the guard silently never runs while the startup log reports it as
    # applied. Mount the same middleware, with the same allow-lists, ourselves.
    #
    # DNS rebinding targets loopback services specifically, so binding 127.0.0.1
    # is not a mitigation and this must apply to the default bind too.
    if transport == "sse":
        protection = security_kwargs.pop("host_origin_protection", "auto")
        allowed_hosts = security_kwargs.pop("allowed_hosts", None)
        allowed_origins = security_kwargs.pop("allowed_origins", None)
        security_kwargs = {}

        if protection is not False:
            extra_middleware.append(
                Middleware(
                    HostOriginGuardMiddleware,
                    allowed_hosts=allowed_hosts,
                    allowed_origins=allowed_origins,
                    mode="strict" if protection is True else "auto",
                )
            )
        else:
            logger.warning(
                "Host/Origin (DNS-rebinding) protection is disabled on the SSE "
                "transport bound to %s.",
                host,
            )

    app = mcp.http_app(
        transport=transport,
        path=path,
        middleware=extra_middleware or None,
        **security_kwargs,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


async def _serve_with_cors(
    transport: str, host: str, port: int, log_level: str, path: str | None = None
):
    """Serve one of FastMCP's HTTP transports under uvicorn.

    FastMCP's own run_http_async() would bind the socket for us but gives no
    seam for the CORS middleware, so we build the ASGI app ourselves.
    """
    app = _build_http_app(transport, host, path)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Session-aware memory operations (remember, recall, forget)
# ---------------------------------------------------------------------------


@registry.tool(tags={DEFAULT_TAG, MEMORY_TAG})
async def remember(
    data: str | None = None,
    filename: str | None = None,
    content_base64: str | None = None,
    dataset_name: str | None = None,
    session_id: str | None = None,
    custom_prompt: str | None = None,
    background: bool = False,
) -> list:
    """Store data in memory.

    Two modes depending on whether session_id is provided:

    Without session_id (permanent memory): Runs the full add + cognify
    pipeline to ingest data and build the knowledge graph.

    With session_id (session memory): Stores the data in the session
    cache only. Fast, no entity extraction. Omit session_id when the
    content should be stored as permanent graph memory.

    Pass either `data` (text) or `filename` + `content_base64` (a file
    upload, up to 10 MB), not both. File uploads are permanent-memory
    only and don't support session_id.

    Parameters
    ----------
    data : str, optional
        The text content to store. Mutually exclusive with
        filename/content_base64.
    filename : str, optional
        Original filename for a file upload. Used to derive the stored
        document's name. Requires content_base64.
    content_base64 : str, optional
        Base64-encoded file content to ingest. Requires filename.
    dataset_name : str, optional
        Target dataset name. Defaults to the current MCP client's
        agent-scoped dataset (e.g. "cursor_vscode_memory"), or
        "main_dataset" if no client identity is detected.
    session_id : str, optional
        Session ID. When set, stores in session cache only.
    custom_prompt : str, optional
        Custom prompt for entity extraction (permanent mode only).
    background : bool
        Queue permanent ingestion as a background task and return immediately
        instead of waiting for the pipeline. Use when the caller has a request
        deadline shorter than ingestion takes. Ignored with session_id, which
        is already fast. Errors surface via cognify_status, not the return
        value.
    """
    if content_base64 and data:
        return [
            types.TextContent(
                type="text",
                text="Error: pass either `data` or `filename` + `content_base64`, not both.",
            )
        ]
    if not content_base64 and not data:
        return [
            types.TextContent(
                type="text",
                text="Error: provide `data` or `filename` + `content_base64`.",
            )
        ]
    if content_base64 and session_id:
        return [
            types.TextContent(
                type="text",
                text="Error: file uploads (content_base64) don't support session_id.",
            )
        ]

    if content_base64:
        try:
            decoded = base64.b64decode(content_base64, validate=True)
        except Exception as e:
            logger.debug("Falling back after error in remember", exc_info=True)
            return [types.TextContent(type="text", text=f"Error: invalid base64 content ({e}).")]
        if len(decoded) > _MAX_UPLOAD_BYTES:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error: file exceeds 10 MB limit ({len(decoded):,} bytes).",
                )
            ]

    dataset_name = dataset_name or _agent_scoped_default_dataset()

    # Permanent-memory ingestion runs add + cognify (+ improve), which routinely
    # outruns an MCP host's per-request deadline — the same constraint the
    # cognify tool documents as "background process launched due to MCP timeout
    # limitations". Callers that can't block pass background=True and poll
    # cognify_status instead. Session-cache writes are fast, so they always
    # run inline.
    if background and not session_id:

        async def remember_task_wrapper(**kwargs):
            """Wrapper that captures errors from the background task."""
            try:
                await cognee_client.remember(**kwargs)
            except Exception as e:
                _record_task_error(dataset_name, str(e))
                logger.exception(f"Background remember task failed for dataset '{dataset_name}'")

        _track_background(
            remember_task_wrapper(
                data=data,
                filename=filename,
                content_base64=content_base64,
                dataset_name=dataset_name,
                session_id=None,
                custom_prompt=custom_prompt,
            )
        )
        queued = f"'{filename}'" if content_base64 else "text"
        return [
            types.TextContent(
                type="text",
                text=(
                    f"Background process launched due to MCP timeout limitations.\n"
                    f"Queued {queued} for dataset '{dataset_name}'.\n"
                    f"Check progress with cognify_status, or the log file at: "
                    f"{get_log_file_location()}"
                ),
            )
        ]

    with redirect_stdout(sys.stderr):
        try:
            result = await cognee_client.remember(
                data=data,
                filename=filename,
                content_base64=content_base64,
                dataset_name=dataset_name,
                session_id=session_id,
                custom_prompt=custom_prompt,
            )
            status = result.get("status", "completed")
            if session_id:
                text = f"Stored in session cache (session_id={session_id}, status={status})."
            elif content_base64:
                text = (
                    f"Ingested '{filename}' ({len(decoded):,} bytes) into dataset "
                    f"'{dataset_name}' (status={status})."
                )
            else:
                text = f"Stored permanently in knowledge graph (dataset={dataset_name}, status={status})."
            return [types.TextContent(type="text", text=text)]
        except Exception as e:
            error_msg = f"Remember failed: {e!s}"
            logger.exception(error_msg)
            return [types.TextContent(type="text", text=f"Error: {error_msg}")]


@registry.tool(tags={DEFAULT_TAG, MEMORY_TAG})
async def recall(
    query: str,
    search_type: str | None = None,
    datasets: str | None = None,
    session_id: str | None = None,
    system_prompt: str | None = None,
    top_k: int = 15,
) -> list:
    """Search memory with auto-routing and session awareness.

    When session_id is provided without datasets or search_type,
    searches session cache first by keyword matching. Falls through
    to the permanent knowledge graph if no session results match.

    Auto-routing picks the best search strategy when search_type
    is not specified.

    Parameters
    ----------
    query : str
        Natural language query to search for.
    search_type : str, optional
        Override auto-routing. Options: GRAPH_COMPLETION,
        GRAPH_COMPLETION_COT, RAG_COMPLETION, CHUNKS, SUMMARIES,
        TEMPORAL, FEELING_LUCKY, etc.
    datasets : str, optional
        Comma-separated dataset names to search within.
    session_id : str, optional
        Session ID for session-first search.
    system_prompt : str, optional
        Override the synthesis prompt for completion searches. When omitted,
        falls back to COGNEE_MCP_RECALL_SYSTEM_PROMPT / _FILE if configured
        on the server.
    top_k : int
        Maximum results to return (default: 10).
    """
    with redirect_stdout(sys.stderr):
        try:
            normalized_top_k = validate_top_k(top_k)
            dataset_list = parse_csv_list(datasets)
            results = await cognee_client.recall(
                query_text=query,
                search_type=search_type,
                datasets=dataset_list,
                session_id=session_id,
                system_prompt=system_prompt,
                top_k=normalized_top_k,
            )
            return [
                types.TextContent(
                    type="text",
                    text=format_recall_results(results, json_encoder=JSONEncoder),
                )
            ]
        except Exception as e:
            error_msg = f"Recall failed: {e!s}"
            logger.exception(error_msg)
            return [types.TextContent(type="text", text=f"Error: {error_msg}")]


@registry.tool(tags={DEFAULT_TAG, MEMORY_TAG})
async def forget(
    dataset: str | None = None,
    everything: bool = False,
    data_id: str | None = None,
    dataset_id: str | None = None,
) -> list:
    """Delete data from memory.

    Can target a single data item, a specific dataset (by name or id), or
    everything the user owns. Removes data from the relational DB, graph DB,
    and vector DB.

    Parameters
    ----------
    dataset : str, optional
        Dataset name to delete entirely.
    everything : bool
        If true, delete ALL data across all datasets.
    data_id : str, optional
        UUID of a single data item to delete. Must be paired with `dataset`
        or `dataset_id` so the owning dataset is unambiguous.
    dataset_id : str, optional
        UUID of the dataset to delete entirely, or to scope `data_id`.
    """
    with redirect_stdout(sys.stderr):
        try:
            if not dataset and not everything and not data_id and not dataset_id:
                return [
                    types.TextContent(
                        type="text",
                        text=(
                            "Error: Specify 'dataset' name or set 'everything' to true. "
                            "To remove a single item, pass 'data_id' with 'dataset' or "
                            "'dataset_id'."
                        ),
                    )
                ]
            if data_id and not dataset and not dataset_id:
                return [
                    types.TextContent(
                        type="text",
                        text="Error: 'data_id' requires 'dataset' or 'dataset_id'.",
                    )
                ]

            # The UI passes ids as strings over JSON; cognee.forget() wants UUIDs.
            # Parse here so a malformed id is a clear message rather than a
            # cognee-internal traceback.
            from uuid import UUID

            try:
                parsed_data_id = UUID(data_id) if data_id else None
                parsed_dataset_id = UUID(dataset_id) if dataset_id else None
            except ValueError as e:
                return [types.TextContent(type="text", text=f"Error: invalid UUID ({e}).")]

            result = await cognee_client.forget(
                dataset=dataset,
                everything=everything,
                data_id=parsed_data_id,
                dataset_id=parsed_dataset_id,
            )
            status = result.get("status", "unknown") if isinstance(result, dict) else "completed"
            if everything:
                text = f"All data deleted (status={status})."
            elif parsed_data_id:
                text = f"Data item '{data_id}' deleted (status={status})."
            else:
                text = f"Dataset '{dataset or dataset_id}' deleted (status={status})."
            return [types.TextContent(type="text", text=text)]
        except Exception as e:
            error_msg = f"Forget failed: {e!s}"
            logger.exception(error_msg)
            return [types.TextContent(type="text", text=f"Error: {error_msg}")]


# ---------------------------------------------------------------------------
# V1 pipeline status tool
# ---------------------------------------------------------------------------


@registry.tool(
    tags={"status"},
    description=(
        "Check the progress of background ingestion started by remember(background=True). "
        "Reports active and completed pipeline jobs for a dataset, including failures that "
        "a backgrounded call could not return inline."
    ),
)
async def cognify_status(
    dataset_name: str | None = None,
    pipelines: list[str] | None = None,
) -> list:
    """
    Get the current status of selected pipelines.

    This function retrieves information about current and recently completed
    pipeline operations in the selected dataset. When `dataset_name` is omitted
    it defaults to the current MCP client's agent-scoped dataset (e.g.
    "cursor_vscode_memory") so each agent sees its own status.

    Returns
    -------
    list
        A list containing a single TextContent object with the status information as a string.
        The status includes information about active and completed jobs for the
        requested pipelines.

    Notes
    -----
    - By default this checks "cognify_pipeline" (backward compatible)
    - Use `pipelines` to restrict to specific pipeline names
    - Status information includes job progress, execution time, and completion status
    - The status is returned in string format for easy reading
    - In API mode the dataset id is resolved over HTTP and status is read
      from the server's `GET /api/v1/datasets/status` endpoint
    """
    dataset_name = dataset_name or _agent_scoped_default_dataset()
    with redirect_stdout(sys.stderr):
        try:
            if cognee_client.use_api:
                # API mode: resolve the dataset id over HTTP (no local cognee
                # instance exists in this process) before querying status.
                datasets = await cognee_client.list_datasets()
                dataset_id = next(
                    (d["id"] for d in datasets if d.get("name") == dataset_name), None
                )
                if dataset_id is None:
                    return [
                        types.TextContent(
                            type="text",
                            text=f"❌ Dataset '{dataset_name}' not found via API",
                        )
                    ]
            else:
                from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
                from cognee.modules.users.methods import get_default_user

                user = await get_default_user()
                dataset_id = await get_unique_dataset_id(dataset_name, user)

            requested_pipelines = list(dict.fromkeys(pipelines or ["cognify_pipeline"]))

            if len(requested_pipelines) == 1:
                status = await cognee_client.get_pipeline_status(
                    [dataset_id], requested_pipelines[0]
                )
            else:
                status: dict[str, dict] = {str(dataset_id): {}}
                for pipeline_name in requested_pipelines:
                    pipeline_status = await cognee_client.get_pipeline_status(
                        [dataset_id], pipeline_name
                    )
                    if str(dataset_id) in pipeline_status:
                        status[str(dataset_id)][pipeline_name] = pipeline_status[str(dataset_id)]

            # Append any background task errors
            status_text = str(status)
            dataset_errors = _task_errors.get(dataset_name, [])
            if dataset_errors:
                error_lines = ["\n\nBackground task errors:"]
                for ts, err in sorted(dataset_errors, reverse=True):
                    error_lines.append(f"  [{ts}] {err}")
                status_text += "\n".join(error_lines)

            return [types.TextContent(type="text", text=status_text)]
        except NotImplementedError:
            error_msg = "❌ Pipeline status is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Failed to get cognify status: {e!s}"
            # Still report background errors even if pipeline status fails
            dataset_errors = _task_errors.get(dataset_name, [])
            if dataset_errors:
                error_lines = ["\n\nBackground task errors:"]
                for ts, err in sorted(dataset_errors, reverse=True):
                    error_lines.append(f"  [{ts}] {err}")
                error_msg += "\n".join(error_lines)
            logger.exception(error_msg)
            return [types.TextContent(type="text", text=error_msg)]


def _sanitize_client_name(name: str) -> str:
    import re

    # Strip parenthetical suffixes that bridges like mcp-remote append, e.g.
    # "cursor-vscode (via mcp-remote 0.1.37)" -> "cursor-vscode".
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", (name or "").strip())
    s = re.sub(r"[^a-z0-9_]+", "_", cleaned.lower()).strip("_")
    return s or "unknown"


def _is_agent_scoping_enabled() -> bool:
    """Whether per-client default datasets are active.

    Controlled by COGNEE_MCP_AGENT_SCOPED env var (default: 'true').
    When 'false', tools fall back to 'main_dataset' as the default,
    matching the pre-agent-scoping behavior.
    """
    return os.getenv("COGNEE_MCP_AGENT_SCOPED", "true").strip().lower() != "false"


def _agent_scoped_default_dataset() -> str:
    """Return the default dataset for the current MCP request.

    With agent scoping enabled (default), reads clientInfo.name from the
    active request context and returns '{sanitized}_memory' (e.g.
    'cursor_vscode_memory'). Falls back to 'main_dataset' when:
      - agent scoping is disabled via COGNEE_MCP_AGENT_SCOPED=false, or
      - no client identity is available on the request.

    Used as the runtime default for tool params like dataset_name so each
    MCP client writes to its own scope automatically when the LLM omits
    the argument.
    """
    if not _is_agent_scoping_enabled():
        return "main_dataset"

    from mcp.server.lowlevel.server import request_ctx

    try:
        ctx = request_ctx.get()
        params = getattr(ctx.session, "client_params", None)
        if params and params.clientInfo and params.clientInfo.name:
            return f"{_sanitize_client_name(params.clientInfo.name)}_memory"
    except LookupError:
        pass
    return "main_dataset"


async def main():
    global cognee_client

    # Operations run in-process by this MCP server record origin="mcp" in
    # pipeline_runs. (In client mode the remote API records origin="api".)
    # Guarded because cognee-mcp depends on cognee from PyPI (see
    # pyproject.toml), which may predate cognee.modules.operations — origin
    # stamping is optional, booting is not. Loud, not silent: the warning
    # names exactly what is degraded and when the guard can be deleted.
    try:
        from cognee.modules.operations import ORIGIN_MCP, set_operation_origin

        set_operation_origin(ORIGIN_MCP)
    except ImportError:
        logger.warning(
            "Installed cognee has no cognee.modules.operations — pipeline_runs "
            "records from this MCP server will show origin='sdk' instead of "
            "'mcp'. Remove this guard once cognee-mcp requires a cognee release "
            "that ships the operations module (SDK-399)."
        )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport",
        choices=["sse", "stdio", "http"],
        default="stdio",
        help="Transport to use for communication with the client. (default: stdio)",
    )

    # HTTP transport options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the HTTP server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server to (default: 8000)",
    )

    parser.add_argument(
        "--path",
        default=None,
        help="Path for the MCP HTTP endpoint. Defaults per transport: /mcp for "
        "http, /sse for sse. Applies to http and sse only; ignored for stdio.",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for the HTTP server (default: info)",
    )

    parser.add_argument(
        "--no-migration",
        default=False,
        action="store_true",
        help="Argument stops database migration from being attempted",
    )

    parser.add_argument(
        "--tool-mode",
        default=None,
        choices=TOOL_MODES,
        help="How many tools to advertise in tools/list. 'default' pins the memory API "
        "and makes the rest discoverable via search_tools; "
        "'minimal' pins only the memory API; 'all' advertises every tool. "
        "Can also be set via COGNEE_MCP_TOOL_MODE. (default: default)",
    )

    # Cognee API connection options
    parser.add_argument(
        "--api-url",
        default=os.getenv("COGNEE_BASE_URL"),
        help="Base URL of a running Cognee FastAPI server or Cognee Cloud tenant "
        "(e.g., https://<tenant>.cognee.ai). If provided, the MCP server connects to the "
        "API instead of using cognee directly. Can also be set via the COGNEE_BASE_URL env var.",
    )

    parser.add_argument(
        "--api-token",
        default=os.getenv("COGNEE_API_KEY"),
        help="Authentication token for the API, sent as X-Api-Key (required if the API has "
        "authentication enabled). Can also be set via the COGNEE_API_KEY env var.",
    )

    # Cognee Cloud connection options
    parser.add_argument(
        "--serve-url",
        default=None,
        help="Cognee Cloud or remote instance URL (e.g., https://your-instance.cognee.ai). "
        "Calls cognee.serve() at startup so all SDK operations route to the cloud. "
        "Can also be set via COGNEE_SERVICE_URL env var.",
    )

    parser.add_argument(
        "--serve-api-key",
        default=None,
        help="API key for the Cognee Cloud instance. Can also be set via COGNEE_API_KEY env var.",
    )

    args = parser.parse_args()

    # Initialize the global CogneeClient
    cognee_client = CogneeClient(api_url=args.api_url, api_token=args.api_token)

    host = args.host
    port = int(args.port)
    apply_tool_mode(args.tool_mode)

    # Resolve cloud connection: CLI args take precedence over env vars
    serve_url = args.serve_url or os.environ.get("COGNEE_SERVICE_URL", "")
    serve_api_key = args.serve_api_key or os.environ.get("COGNEE_API_KEY", "")

    # Connect to Cognee Cloud if configured (before migrations — cloud handles its own DB)
    if serve_url and not args.api_url:
        import cognee

        serve_kwargs = {"url": serve_url}
        if serve_api_key:
            serve_kwargs["api_key"] = serve_api_key
        await cognee.serve(**serve_kwargs)
        logger.info(f"Connected to Cognee Cloud: {serve_url}")

    # Skip migrations when in API or Cloud mode (remote handles its own database)
    is_remote = bool(args.api_url) or bool(serve_url)
    if not args.no_migration and not is_remote:
        from cognee.run_migrations import run_migrations

        logger.info("Running database migrations...")

        # Migrations print progress and "table already exists" notices to
        # stdout. In stdio transport stdout is the JSON-RPC channel, so route
        # that output to stderr — the same guard every tool applies around its
        # cognee calls.
        with redirect_stdout(sys.stderr):
            # run_migrations() alone — it tells a fresh database from an
            # existing one (fresh: schema from the models + `alembic stamp
            # head`; existing: Alembic deltas + the graph/vector data chain).
            # Running setup() before it used to spoil that check: create_all
            # built the schema unstamped, so a brand-new database was
            # classified as existing and replayed the full migration history.
            # Vector-store tables are created on first write (add() runs
            # setup()), the same as every SDK flow.
            await run_migrations()

        logger.info("Database migrations done.")
    elif not is_remote:
        logger.info("Skipping DB migrations")

    try:
        match args.transport.lower():
            case "sse":
                sse_path = args.path or fastmcp.settings.sse_path
                logger.info(f"Running MCP server with SSE transport on {host}:{port}{sse_path}")
                await _serve_with_cors("sse", host, port, args.log_level, args.path)
            case "http":
                http_path = args.path or fastmcp.settings.streamable_http_path
                logger.info(
                    f"Running MCP server with Streamable HTTP transport on {host}:{port}{http_path}"
                )
                await _serve_with_cors("http", host, port, args.log_level, args.path)
            case _:
                logger.info("Running MCP server with stdio")
                # show_banner=False: the banner is cosmetic and its version check
                # makes a network call on every startup.
                await mcp.run_stdio_async(show_banner=False)
    finally:
        # Drain background tasks with a bounded timeout so a hung cognify can't
        # block shutdown indefinitely. Then close the HTTP client pool.
        if _background_tasks:
            logger.info(f"Awaiting {len(_background_tasks)} background task(s) before shutdown")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*_background_tasks, return_exceptions=True),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"{len(_background_tasks)} background task(s) still running at shutdown; cancelling"
                )
                for t in _background_tasks:
                    t.cancel()
        if cognee_client is not None:
            await cognee_client.close()


if __name__ == "__main__":
    logger = setup_logging()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee MCP server: {e!s}")
        raise
