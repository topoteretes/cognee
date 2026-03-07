from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from .Document import Document


class DltRowDocument(Document):
    """Document type for DLT-ingested relational rows.

    Skips chunking entirely — yields a single DocumentChunk containing
    the enriched schema context text. The graph for DLT data is built
    deterministically by extract_dlt_fk_edges, so no LLM extraction
    is needed.
    """

    type: str = "dlt_row"
    mime_type: str = "application/x-dlt-row"

    async def read(self, chunker_cls: Chunker, max_chunk_size: int):
        from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
        from cognee.infrastructure.files.utils.open_data_file import open_data_file

        async with open_data_file(self.raw_data_location, mode="r", encoding="utf-8") as file:
            text = file.read()

        if not text or not text.strip():
            return

        yield DocumentChunk(
            id=generate_node_id(f"{self.id}_chunk_0"),
            text=text.strip(),
            chunk_size=len(text.strip().split()),
            chunk_index=0,
            cut_type="dlt_row",
            is_part_of=self,
            contains=[],
        )
