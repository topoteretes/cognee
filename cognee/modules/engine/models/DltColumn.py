from cognee.modules.engine.models.ColumnValue import ColumnValue


class DltColumn(ColumnValue):
    """A shared cell-value node emitted by the DLT pipeline.

    Subclasses ColumnValue (the relational-migration pipeline's value node)
    so generic machinery treats both as one family via isinstance, while the
    graph type name distinguishes DLT-emitted values for type-name filtering
    alongside DltRow and the schema nodes. Ids stay uuid5-seeded from
    (table, column, value), so identity is unchanged.
    """
