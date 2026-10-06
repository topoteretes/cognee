"""Store exact-name ``UUID`` columns as text on SQLite.

SQLite derives a column's type affinity from its declared type name. ``UUID``
matches none of its rules, so such a column gets NUMERIC affinity and a value
made only of digits (a hex UUID can be) is stored as a number, losing digits
and breaking equality against the text form. SQLAlchemy's portable ``Uuid``
renders ``CHAR(32)`` on SQLite (TEXT affinity); the exact-name ``UUID`` type
renders ``UUID``.

The models declare ``Uuid``. The frozen base schema and the Alembic chain
that build every fresh database still declare ``sa.UUID()`` in their
historical revisions, which are not rewritten, so this hook makes ``UUID``
compile the way ``Uuid`` does on SQLite. Postgres is untouched and keeps its
native type. Importing this module registers the hook; the relational
package imports it, so every engine and the Alembic env get it.
"""

from sqlalchemy import UUID
from sqlalchemy.ext.compiler import compiles


@compiles(UUID, "sqlite")
def compile_uuid_as_text_on_sqlite(type_, compiler, **kw) -> str:
    """Render ``UUID`` as ``Uuid`` would on SQLite: ``CHAR(32)``."""
    return compiler.visit_uuid(type_, **kw)
