from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


class DltRow(DocumentChunk):
    """One row of a DLT source manifest — its own graph type AND its own
    vector collection (DltRow_text).

    Chunk search (DocumentChunk_text) is for document ingestion only, so rows
    deliberately do NOT appear there. Row text is searchable through the
    collections that opt in — graph completion includes DltRow_text,
    and the row-aware search type reads it directly. Rows carry
    ``cut_type="dlt_row"`` and structural edges (is_row_of → SchemaTable, FK
    edges, column-value edges). Subclassing DocumentChunk reuses the chunk
    field shape; filter by type NAME, not isinstance.
    """
