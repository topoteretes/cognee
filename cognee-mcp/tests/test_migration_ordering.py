"""Regression tests for fresh-DB / remote Docker migration ordering in the MCP server.

Context (GitHub issues #1931, #2007, #2022):

After table creation was moved out of Alembic into
``SqlAlchemyAdapter.create_database()`` / ``create_all()``, entrypoints that ran
``alembic upgrade head`` before the tables existed crash-looped on fresh
databases (``NoSuchTableError: acls``; ``type "pipelinerunstatus" does not
exist``). PR #1932 added a ``setup.py`` create-all fallback to the main
``entrypoint.sh`` but NOT to ``cognee-mcp/entrypoint.sh`` (#2007), so the MCP
container kept crash-looping.

The current fix moves the ordering INTO ``cognee-mcp/src/server.py:main()``:
``setup()`` (which creates all tables) is called BEFORE ``run_migrations()``,
and both are skipped in API / Cloud (remote) modes or when ``--no-migration``
is passed. The shell entrypoint no longer touches the database.

These tests pin that behaviour so a refactor cannot silently reintroduce the
crash-loop by dropping the ``setup()`` call, reordering it after the migration,
or running migrations in remote mode.
"""

import ast
import importlib
from pathlib import Path

MCP_SERVER = Path(__file__).resolve().parents[1] / "src" / "server.py"


def _find_main(tree: ast.Module) -> ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "main":
            return node
    raise AssertionError("Could not find `async def main()` in cognee-mcp/src/server.py")


def _await_call_names(node: ast.AST) -> list[str]:
    """Return, in source order, the names of awaited function calls under `node`.

    e.g. ``await setup()`` -> "setup", ``await run_migrations()`` -> "run_migrations".
    """
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Await) and isinstance(child.value, ast.Call):
            func = child.value.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


def _migration_branch(main_node: ast.AsyncFunctionDef) -> ast.If:
    """Locate the `if not args.no_migration and not is_remote:` migration branch."""
    for node in ast.walk(main_node):
        if isinstance(node, ast.If):
            calls = _await_call_names(node)
            if "setup" in calls and "run_migrations" in calls:
                return node
    raise AssertionError(
        "Could not find the migration branch in main() that awaits both "
        "setup() and run_migrations(). The #2007 create-all-before-migrate fix "
        "may have been removed."
    )


def test_mcp_server_setup_runs_before_migrations():
    """setup() (create_all) must run BEFORE run_migrations() to avoid #2007 crash-loop."""
    tree = ast.parse(MCP_SERVER.read_text(encoding="utf-8"))
    main_node = _find_main(tree)
    branch = _migration_branch(main_node)

    calls = _await_call_names(branch)
    assert calls.index("setup") < calls.index("run_migrations"), (
        "cognee-mcp must call setup() (table create_all) BEFORE run_migrations(); "
        f"got await order {calls!r}. Running alembic on a fresh DB before tables "
        "exist reintroduces the crash-loop from issues #1931/#2007/#2022."
    )


def test_mcp_server_migration_guarded_against_remote_mode():
    """Migrations must be skipped in remote (API/Cloud) and --no-migration modes."""
    source = MCP_SERVER.read_text(encoding="utf-8")
    # The remote-mode guard and the no-migration flag must both gate the branch.
    assert "is_remote" in source, "Lost the remote-mode guard around DB migrations (#2007)."
    assert "no_migration" in source, "Lost the --no-migration guard around DB migrations."

    tree = ast.parse(source)
    main_node = _find_main(tree)
    branch = _migration_branch(main_node)
    test_src = ast.unparse(branch.test)
    assert "no_migration" in test_src and "is_remote" in test_src, (
        "The migration branch must be guarded by both `not args.no_migration` and "
        f"`not is_remote`; got guard `{test_src}`."
    )


def test_setup_and_run_migrations_are_importable():
    """The exact symbols cognee-mcp imports for the fix must exist and be importable.

    Guards against the fix breaking if these modules are moved/renamed.
    """
    setup_mod = importlib.import_module("cognee.modules.engine.operations.setup")
    migrations_mod = importlib.import_module("cognee.run_migrations")
    assert callable(setup_mod.setup)
    assert callable(migrations_mod.run_migrations)


def test_mcp_entrypoint_does_not_run_alembic():
    """The shell entrypoint must NOT run migrations itself (server.py owns ordering).

    If alembic creeps back into the entrypoint it would run before the in-process
    setup() create-all, reintroducing the ordering bug the way #2007 originally hit.
    """
    entrypoint = MCP_SERVER.resolve().parents[1] / "entrypoint.sh"
    text = entrypoint.read_text(encoding="utf-8")
    assert "alembic upgrade" not in text, (
        "cognee-mcp/entrypoint.sh must not run `alembic upgrade`; database setup and "
        "migration ordering are owned by cognee-mcp/src/server.py:main() (#2007)."
    )
