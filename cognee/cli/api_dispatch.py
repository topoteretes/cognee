"""Dispatch CLI commands via the Cognee HTTP API instead of in-process.

When ``--api-url`` is set, this module intercepts the normal in-process
execution and forwards the command to a running API server.  This ensures
a single process owns all file-based database connections.
"""

from __future__ import annotations

import argparse
import json
import os

import cognee.cli.echo as fmt
from cognee.cli.api_client import CogneeApiClient

SUPPORTED_COMMANDS = {
    "add",
    "cognify",
    "search",
    "memify",
    "datasets",
    "delete",
    "remember",
    "recall",
    "improve",
    "forget",
}


def can_dispatch(args: argparse.Namespace) -> bool:
    """Return True if the command should be dispatched via HTTP."""
    return bool(getattr(args, "api_url", None))


def dispatch(args: argparse.Namespace) -> None:
    """Execute the CLI command by forwarding it to the API server."""
    command = args.command
    user_id = getattr(args, "user_id", None)

    # Build optional auth header so the server can identify the caller.
    headers: dict[str, str] = {}
    if user_id:
        headers["X-User-Id"] = user_id
        fmt.note(
            f"Passing --user-id {user_id} to the API server via X-User-Id header.  "
            f"The server must be configured to honour this header for isolation to work."
        )

    api_key = getattr(args, "api_key", None) or os.environ.get("COGNEE_API_KEY")
    api_token = getattr(args, "api_token", None) or os.environ.get("COGNEE_API_TOKEN")
    if api_key:
        headers["X-Api-Key"] = api_key
    elif api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    with CogneeApiClient(args.api_url, headers=headers) as client:
        # Health probe — fail fast with a clear message
        try:
            client.health()
        except Exception:
            raise RuntimeError(
                f"Cannot connect to Cognee API at {args.api_url}.  "
                f"Is the server running?  Start it with:  "
                f"uvicorn cognee.api.client:app --port 8000"
            )

        dispatchers = {
            "add": _dispatch_add,
            "cognify": _dispatch_cognify,
            "search": _dispatch_search,
            "memify": _dispatch_memify,
            "datasets": _dispatch_datasets,
            "delete": _dispatch_delete,
            "remember": _dispatch_remember,
            "recall": _dispatch_recall,
            "improve": _dispatch_improve,
            "forget": _dispatch_forget,
        }

        handler = dispatchers.get(command)
        if handler is None:
            raise RuntimeError(
                f"Command '{command}' is not supported in --api-url mode "
                f"(supported: {', '.join(sorted(dispatchers))}).  "
                f"Run without --api-url to execute it locally."
            )

        handler(client, args)


# -- individual dispatchers -----------------------------------------------


def _dispatch_add(client: CogneeApiClient, args: argparse.Namespace) -> None:
    fmt.echo(f"Adding {len(args.data)} item(s) to dataset '{args.dataset_name}'...")
    result = client.add(args.data, args.dataset_name)
    fmt.success(f"Successfully added data to dataset '{args.dataset_name}'")
    if result:
        fmt.echo(json.dumps(result, indent=2, default=str))


def _dispatch_cognify(client: CogneeApiClient, args: argparse.Namespace) -> None:
    datasets = args.datasets if args.datasets else None
    fmt.echo("Starting cognification...")
    result = client.cognify(
        datasets=datasets,
        run_in_background=getattr(args, "background", False),
        chunks_per_batch=getattr(args, "chunks_per_batch", None),
    )
    if getattr(args, "background", False):
        fmt.success("Cognification started in background!")
    else:
        fmt.success("Cognification completed successfully!")
    if result:
        fmt.echo(json.dumps(result, indent=2, default=str))


def _dispatch_search(client: CogneeApiClient, args: argparse.Namespace) -> None:
    fmt.echo(f"Searching for: '{args.query_text}'...")
    results = client.search(
        query=args.query_text,
        search_type=args.query_type,
        datasets=args.datasets,
        top_k=args.top_k,
    )

    output_format = getattr(args, "output_format", "pretty")
    if output_format == "json":
        fmt.echo(json.dumps(results, indent=2, default=str))
    elif not results:
        fmt.warning("No results found for your query.")
    else:
        fmt.echo(f"\nFound {len(results)} result(s):")
        fmt.echo("=" * 60)
        for i, result in enumerate(results, 1):
            fmt.echo(f"{fmt.bold(f'Result {i}:')} {result}")
            fmt.echo()


def _dispatch_memify(client: CogneeApiClient, args: argparse.Namespace) -> None:
    label = args.dataset_id or args.dataset_name
    fmt.echo(f"Running memify on '{label}'...")
    result = client.memify(
        dataset_name=args.dataset_name,
        dataset_id=args.dataset_id,
        data=getattr(args, "data", None),
        node_name=getattr(args, "node_name", None),
        run_in_background=getattr(args, "background", False),
    )
    if getattr(args, "background", False):
        fmt.success("Memify started in background.")
    else:
        fmt.success("Memify completed.")
    if result:
        fmt.echo(json.dumps(result, indent=2, default=str))


def _dispatch_datasets(client: CogneeApiClient, args: argparse.Namespace) -> None:
    action = getattr(args, "datasets_action", None)
    if not action:
        fmt.error("No action specified.")
        return

    if action == "list":
        ds = client.datasets_list()
        if not ds:
            fmt.echo("No datasets found.")
            return
        fmt.echo(f"{'ID':<38} {'Name':<30} {'Created'}")
        fmt.echo("-" * 90)
        for d in ds:
            fmt.echo(f"{d['id']:<38} {d['name']:<30} {d.get('created_at', '')}")

    elif action == "create":
        result = client.datasets_create(args.name)
        fmt.success(f"Created dataset '{args.name}' ({result.get('id', '')})")

    elif action == "data":
        items = client.datasets_data(args.dataset_id)
        if not items:
            fmt.echo("No data items found.")
            return
        fmt.echo(f"{'ID':<38} {'Name':<30} {'Type':<15} {'Created'}")
        fmt.echo("-" * 110)
        for d in items:
            fmt.echo(
                f"{d['id']:<38} {d['name']:<30} {d.get('mime_type', ''):<15} {d.get('created_at', '')}"
            )

    elif action == "status":
        statuses = client.datasets_status(
            args.dataset_ids, pipelines=getattr(args, "pipelines", None)
        )
        if not statuses:
            fmt.echo("No status information available.")
            return
        for ds_id, pipeline_statuses in statuses.items():
            if isinstance(pipeline_statuses, dict):
                if not pipeline_statuses:
                    fmt.echo(f"{ds_id}: <no pipeline runs found>")
                    continue
                formatted = ", ".join(
                    f"{pipeline}={status}" for pipeline, status in pipeline_statuses.items()
                )
                fmt.echo(f"{ds_id}: {formatted}")
            else:
                fmt.echo(f"{ds_id}: {pipeline_statuses}")

    elif action == "graph":
        graph = client.datasets_graph(args.dataset_id)
        output = json.dumps(graph, indent=2, default=str)
        if getattr(args, "output", None):
            with open(args.output, "w") as f:
                f.write(output)
            fmt.success(f"Graph exported to {args.output}")
        else:
            fmt.echo(output)

    elif action == "delete":
        if not getattr(args, "force", False):
            if not fmt.confirm(f"Delete dataset {args.dataset_id}? This cannot be undone"):
                fmt.echo("Cancelled.")
                return
        client.datasets_delete(args.dataset_id)
        fmt.success(f"Dataset {args.dataset_id} deleted.")


def _dispatch_delete(client: CogneeApiClient, args: argparse.Namespace) -> None:
    if getattr(args, "all", False):
        if not getattr(args, "force", False):
            if not fmt.confirm("Delete ALL data?"):
                fmt.echo("Cancelled.")
                return
        client.datasets_delete_all()
        fmt.success("All data deleted.")
    elif getattr(args, "dataset_name", None):
        # Resolve dataset name → ID via the list endpoint, then delete
        ds = client.datasets_list()
        match = [d for d in ds if d["name"] == args.dataset_name]
        if not match:
            fmt.error(f"No dataset found with name '{args.dataset_name}'.")
            return
        if not getattr(args, "force", False):
            if not fmt.confirm(f"Delete dataset '{args.dataset_name}'?"):
                fmt.echo("Cancelled.")
                return
        client.datasets_delete(match[0]["id"])
        fmt.success(f"Dataset '{args.dataset_name}' deleted.")
    else:
        fmt.error("Specify --dataset-name or --all for deletion.")


def _dispatch_remember(client: CogneeApiClient, args: argparse.Namespace) -> None:
    fmt.echo(f"Remembering {len(args.data)} item(s) in dataset '{args.dataset_name}'...")
    if getattr(args, "chunker", "TextChunker") != "TextChunker":
        fmt.warning(
            f"--chunker={args.chunker} is ignored in --api-url mode; "
            f"the server selects the chunker."
        )
    result = client.remember(
        data_items=args.data,
        dataset_name=args.dataset_name,
        chunk_size=getattr(args, "chunk_size", None),
        chunks_per_batch=getattr(args, "chunks_per_batch", None),
        run_in_background=getattr(args, "background", False),
    )
    if getattr(args, "background", False):
        fmt.success("Data ingested and cognification started in background!")
    else:
        fmt.success("Data ingested and knowledge graph built successfully!")
    if isinstance(result, dict):
        for key in ("dataset_id", "items_processed", "content_hash", "elapsed_seconds"):
            value = result.get(key)
            if value is not None:
                fmt.echo(f"  {key}: {value}")


def _dispatch_recall(client: CogneeApiClient, args: argparse.Namespace) -> None:
    # Session-only mode: -s without -d and without explicit -t. Mirrors the
    # local recall_command behaviour so --api-url users get the same UX.
    session_only = (
        args.session_id is not None and not args.datasets and args.query_type == "GRAPH_COMPLETION"
    )

    if session_only:
        fmt.echo(f"Searching session '{args.session_id}': '{args.query_text}'")
    else:
        datasets_msg = f" in datasets {args.datasets}" if args.datasets else " across all datasets"
        fmt.echo(f"Recalling: '{args.query_text}' (type: {args.query_type}){datasets_msg}")

    results = client.recall(
        query=args.query_text,
        search_type=None if session_only else args.query_type,
        datasets=args.datasets,
        top_k=args.top_k,
        system_prompt=getattr(args, "system_prompt", None),
        session_id=args.session_id,
    )

    output_format = getattr(args, "output_format", "pretty")
    if output_format == "json":
        fmt.echo(json.dumps(results, indent=2, default=str))
        return
    if output_format == "simple":
        for i, result in enumerate(results, 1):
            fmt.echo(f"{i}. {result}")
        return

    if not results:
        fmt.warning("No results found for your query.")
        return

    is_session = isinstance(results[0], dict) and results[0].get("_source") == "session"
    if is_session:
        fmt.echo(f"\nFound {len(results)} session entry(ies):")
        fmt.echo("=" * 60)
        for i, entry in enumerate(results, 1):
            q = entry.get("question", "")
            a = entry.get("answer", "")
            t = entry.get("time", "")
            header = f"[{t}] " if t else ""
            if q:
                fmt.echo(f"{fmt.bold(f'{header}Q:')} {q}")
            if a:
                fmt.echo(f"{fmt.bold('A:')} {a}")
            if i < len(results):
                fmt.echo("-" * 40)
    else:
        fmt.echo(f"\nFound {len(results)} result(s) using {args.query_type}:")
        fmt.echo("=" * 60)
        if args.query_type in ["GRAPH_COMPLETION", "RAG_COMPLETION"]:
            for i, result in enumerate(results, 1):
                fmt.echo(f"{fmt.bold('Response:')} {result}")
                if i < len(results):
                    fmt.echo("-" * 40)
        elif args.query_type == "CHUNKS":
            for i, result in enumerate(results, 1):
                fmt.echo(f"{fmt.bold(f'Chunk {i}:')} {result}")
                fmt.echo()
        else:
            for i, result in enumerate(results, 1):
                fmt.echo(f"{fmt.bold(f'Result {i}:')} {result}")
                fmt.echo()


def _dispatch_improve(client: CogneeApiClient, args: argparse.Namespace) -> None:
    dataset = args.dataset_id or args.dataset_name
    fmt.echo(f"Improving knowledge graph for dataset '{dataset}'...")
    if getattr(args, "feedback_alpha", 0.1) != 0.1:
        fmt.warning("--feedback-alpha is ignored in --api-url mode; the server uses its default.")
    result = client.improve(
        dataset_name=args.dataset_name if not args.dataset_id else None,
        dataset_id=args.dataset_id,
        node_name=getattr(args, "node_name", None),
        session_ids=getattr(args, "session_ids", None),
        run_in_background=getattr(args, "background", False),
    )
    if getattr(args, "background", False):
        fmt.success("Improvement started in background!")
    else:
        fmt.success("Knowledge graph improved successfully!")
    if result:
        fmt.echo(json.dumps(result, indent=2, default=str))


def _dispatch_forget(client: CogneeApiClient, args: argparse.Namespace) -> None:
    everything = getattr(args, "everything", False)
    dataset = getattr(args, "dataset", None)
    data_id = getattr(args, "data_id", None)
    if not everything and not dataset and not data_id:
        fmt.error("Specify --dataset, --data-id with --dataset, or --everything.")
        return
    result = client.forget(
        dataset=dataset,
        data_id=data_id,
        everything=everything,
    )
    fmt.success(f"Done: {result}")
