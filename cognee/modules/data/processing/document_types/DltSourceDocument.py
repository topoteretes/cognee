from os.path import basename
from uuid import UUID

from cognee.modules.chunking.Chunker import Chunker
from .Document import Document


class DltSourceDocument(Document):
    """Document type for a whole DLT-ingested source (one Data item per source).

    The raw data is a JSON manifest describing every unique row of the source
    (see resolve_dlt_sources._build_source_manifest_item). Skips text chunking
    entirely — yields one DocumentChunk per manifest row so each row remains an
    individual node in the graph/vector stores. Chunk ids are the stable row
    node ids from the manifest, which FK edges reference. The graph structure
    is built deterministically by extract_dlt_source_edges, so no LLM
    extraction is needed.
    """

    type: str = "dlt_source"
    mime_type: str = "application/x-dlt-source"

    async def read(self, chunker_cls: Chunker, max_chunk_size: int):
        from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
        from cognee.tasks.ingestion.dlt_utils import load_dlt_manifest

        manifest = await load_dlt_manifest(self.raw_data_location)

        for chunk_index, row in enumerate(manifest.get("rows", [])):
            text = (row.get("text") or "").strip()
            if not text:
                continue

            yield DocumentChunk(
                id=UUID(row["node_id"]),
                text=text,
                chunk_size=len(text.split()),
                chunk_index=chunk_index,
                cut_type="dlt_row",
                is_part_of=self,
                contains=[],
                document_id=str(self.id),
                document_name=self.name or basename(self.raw_data_location),
            )
