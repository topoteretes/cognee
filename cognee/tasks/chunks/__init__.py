from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from .chunk_by_word import chunk_by_word
from .chunk_by_sentence import chunk_by_sentence
from .chunk_by_paragraph import chunk_by_paragraph
from .remove_disconnected_chunks import remove_disconnected_chunks

# Instantiate retriever
chunks_retriever = ChunksRetriever()


# Define async functions to expose retrieval functionality
async def query_chunks(query: str):
    return await chunks_retriever.get_completion(query)
