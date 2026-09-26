"""Driver availability for the Turso backends. Imports nothing from pyturso itself."""

INSTALL_HINT = (
    "Turso dependencies are not installed. Install them with "
    "'pip install cognee\"[turso]\"' to use the Turso backends."
)


def require_turso():
    """Import and return the ``turso`` module, or raise a cognee-worded ImportError."""
    try:
        import turso
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    return turso
