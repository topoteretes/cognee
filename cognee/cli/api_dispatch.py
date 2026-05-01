"""Dispatch CLI commands via the Cognee HTTP API instead of in-process.

When ``--api-url`` is set, this module intercepts the normal in-process
execution and forwards the command to a running API server.  This ensures
a single process owns all file-based database connections.
"""

from __future__ import annotations

import argparse
import json

import cognee.cli.echo as fmt
from cognee.cli.api_client import CogneeApiClient

SUPPORTED_COMMANDS = {
    "add",
    "cognify",
    "search",
    "memify",
    "datasets",
    "delete",
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
