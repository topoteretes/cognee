"""Compatibility shim — migration orchestration lives in
``cognee.modules.migrations.startup``.

This module is imported by ``cognee/__init__.py`` at package-import time, so
it must stay dependency-free: the real implementations are imported lazily
inside each function (the modules tree pulls in ORM models and database
factories that cannot load during package import).
"""


async def run_relational_migrations():
    """Relational (Alembic) schema migrations only. See ``modules.migrations.startup``."""
    from cognee.modules.migrations.startup import run_relational_migrations

    return await run_relational_migrations()


async def run_migrations():
    """All migrations (relational + graph/vector revision chains). See
    ``modules.migrations.startup``."""
    from cognee.modules.migrations.startup import run_migrations as _run

    return await _run()
