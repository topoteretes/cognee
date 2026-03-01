from uuid import uuid5
from cognee.modules.chunking.models import DocumentChunk
from cognee.shared.data_models import SummarizedContent
from cognee.tasks.summarization.models import TextSummary


def extract_summary(document_chunk: DocumentChunk, summary=SummarizedContent) -> TextSummary:
    return TextSummary(
        id=uuid5(document_chunk.id, "TextSummary"),
        text=summary.summary,
        made_from=document_chunk,
    )
