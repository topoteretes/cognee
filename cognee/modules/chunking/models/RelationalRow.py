from typing import ClassVar

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


class RelationalRow(DocumentChunk):
    """One relational row from a DLT source manifest — its own graph type.

    Graph nodes carry type "RelationalRow", so type-NAME filtering (census,
    visualization, row-aware search) separates relational rows from prose
    chunks. ``index_type_name`` keeps its vectors in the DocumentChunk_text
    collection, so every existing retriever finds rows without DLT-specific
    code. NOTE for search implementers: subclassing DocumentChunk is the
    mechanism for that shared collection — filter by type name, not
    isinstance, or rows will be swept up with prose chunks. Rows additionally
    carry ``cut_type="dlt_row"`` and structural edges (is_row_of →
    SchemaTable, FK edges, column-value edges).
    """

    index_type_name: ClassVar[str] = "DocumentChunk"
