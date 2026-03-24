"""Interactive REPL for querying the Cognee knowledge graph."""

import asyncio
import json
import sys

import cognee.cli.echo as fmt
from cognee.cli.session import load_session, save_session


REPL_COMMANDS = {
    "/help": "Show available commands",
    "/dataset <name>": "Switch active dataset",
    "/type <type>": "Switch query type (e.g. GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS)",
    "/status": "Show current session settings",
    "/quit": "Exit interactive mode",
}


def run_interactive(session: dict) -> int:
    """Run the interactive REPL loop.

    Returns an exit code (0 = clean exit, 1 = error).
    """
    dataset = session.get("dataset", "main_dataset")
    query_type = session.get("query_type", "GRAPH_COMPLETION")

    fmt.echo("Cognee interactive mode")
    fmt.echo(f"  Dataset:    {dataset}")
    fmt.echo(f"  Query type: {query_type}")
    fmt.echo("Type a query, or /help for commands. Ctrl-D to exit.\n")

    # Enable readline history if available
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    # Track the last successfully used dataset/query_type for session save
    last_good_dataset = dataset
    last_good_query_type = query_type

    while True:
        try:
            line = input("cognee> ").strip()
        except (EOFError, KeyboardInterrupt):
            fmt.echo("")
            break

        if not line:
            continue

        # Handle slash commands
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/help":
                for name, desc in REPL_COMMANDS.items():
                    fmt.echo(f"  {name:20s} {desc}")
                continue
            elif cmd == "/dataset":
                if arg:
                    dataset = arg
                    fmt.success(f"Dataset: {dataset}")
                else:
                    fmt.echo(f"Dataset: {dataset}")
                continue
            elif cmd == "/type":
                if arg:
                    query_type = arg.upper()
                    fmt.success(f"Query type: {query_type}")
                else:
                    fmt.echo(f"Query type: {query_type}")
                continue
            elif cmd == "/status":
                fmt.echo(f"  Dataset:    {dataset}")
                fmt.echo(f"  Query type: {query_type}")
                continue
            else:
                fmt.warning(f"Unknown command: {cmd}. Type /help for available commands.")
                continue

        # Execute query
        try:
            results = _run_query(line, query_type, dataset)

            # Query succeeded — update last-known-good state
            last_good_dataset = dataset
            last_good_query_type = query_type

            if not results:
                fmt.warning("No results found.")
            elif query_type in ("GRAPH_COMPLETION", "RAG_COMPLETION"):
                for result in results:
                    fmt.echo(str(result))
            else:
                for i, result in enumerate(results, 1):
                    fmt.echo(f"{i}. {result}")
            fmt.echo("")
        except Exception as e:
            fmt.error(f"Query failed: {e}")

    # Only save the last successfully used settings
    save_session(dataset=last_good_dataset, query_type=last_good_query_type)
    fmt.note("Session saved. Use 'cognee-cli -c' to resume.")
    return 0


def run_prompt(prompt: str, session: dict) -> dict:
    """One-shot query mode. Returns a result dict for JSON envelope.

    Always returns a dict; the caller decides how to display it.
    """
    dataset = session.get("dataset", "main_dataset")
    query_type = session.get("query_type", "GRAPH_COMPLETION")

    results = _run_query(prompt, query_type, dataset)

    # In human mode, print results
    if not fmt.is_json_mode():
        if not results:
            fmt.warning("No results found.")
        elif query_type in ("GRAPH_COMPLETION", "RAG_COMPLETION"):
            for result in results:
                fmt.echo(str(result))
        else:
            for i, result in enumerate(results, 1):
                fmt.echo(f"{i}. {result}")

    return {
        "results": results,
        "query_type": query_type,
        "dataset": dataset,
        "count": len(results) if results else 0,
    }


def _run_query(query_text: str, query_type_str: str, dataset: str = None) -> list:
    """Execute a search query synchronously.

    Uses asyncio.run() when no loop is running (normal CLI usage).
    Falls back to loop.run_until_complete() if called from an existing loop.
    """
    import cognee
    from cognee.modules.search.types import SearchType

    search_type = SearchType[query_type_str]

    async def _search():
        return await cognee.search(
            query_text=query_text,
            query_type=search_type,
            datasets=[dataset] if dataset else None,
        )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _search()).result()
    return asyncio.run(_search())
