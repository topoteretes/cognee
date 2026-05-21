"""Tests for legacy Kuzu compatibility shims."""


def test_kuzu_import_shim_points_to_ladybug():
    from cognee.infrastructure.databases.graph.kuzu.kuzu_migrate import kuzu_migration
    import kuzu
    import kuzu.database
    import ladybug
    import ladybug.database

    assert kuzu.__version__ == ladybug.__version__
    assert kuzu.Database is ladybug.database.Database
    assert kuzu.database.Database is ladybug.database.Database
    assert callable(kuzu_migration)
