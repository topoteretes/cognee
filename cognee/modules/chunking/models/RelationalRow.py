from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


class RelationalRow(DocumentChunk):
    """One relational row from a DLT source manifest — its own graph type.

    Graph nodes carry type "RelationalRow", so type-NAME filtering (census,
    visualization, row-aware search) separates relational rows from prose
    chunks. ``metadata["index_type_name"]`` keeps its vectors in the
    DocumentChunk_text collection, so every existing retriever finds rows
    without DLT-specific code. It lives in metadata (an instance FIELD, not a
    class attribute) because the pipeline rebuilds data-point classes via
    copy_model/create_model, which preserves only the class NAME and fields —
    anything attached to the class object is silently lost. NOTE for search
    implementers: filter by type name, not isinstance. Rows additionally
    carry ``cut_type="dlt_row"`` and structural edges (is_row_of →
    SchemaTable, FK edges, column-value edges).
    """

    metadata: dict = {"index_fields": ["text"], "index_type_name": "DocumentChunk"}
