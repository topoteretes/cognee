"""Tests for legacy Kuzu compatibility shims."""


def test_kuzu_import_shim_points_to_ladybug():
    import ladybug
    import ladybug.database

    import kuzu
    import kuzu.database
    from cognee.infrastructure.databases.graph.kuzu.kuzu_migrate import kuzu_migration

    assert kuzu.__version__ == ladybug.__version__
    assert kuzu.Database is ladybug.database.Database
    assert kuzu.database.Database is ladybug.database.Database
    assert callable(kuzu_migration)
