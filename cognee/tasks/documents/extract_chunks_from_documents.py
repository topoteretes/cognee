from cognee.modules.data.processing.document_types.Document import Document


async def extract_chunks_from_documents(documents: list[Document], chunk_size: int = 1024):
    for document in documents:
        for document_chunk in document.read(chunk_size = chunk_size):
            yield document_chunk
