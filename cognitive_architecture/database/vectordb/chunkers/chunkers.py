"""Module for chunking text data based on various strategies."""

import re
import logging
from cognitive_architecture.database.vectordb.chunkers.chunk_strategy import ChunkStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_data(chunk_strategy=None, source_data=None, chunk_size=None, chunk_overlap=None):
    """Chunk the given source data into smaller parts based on the specified strategy."""
    if chunk_strategy == ChunkStrategy.VANILLA:
        chunked_data = vanilla_chunker(source_data, chunk_size, chunk_overlap)
    elif chunk_strategy == ChunkStrategy.PARAGRAPH:
        chunked_data = chunk_data_by_paragraph(source_data, chunk_size, chunk_overlap)
    elif chunk_strategy == ChunkStrategy.SENTENCE:
        chunked_data = chunk_by_sentence(source_data, chunk_size, chunk_overlap)
    elif chunk_strategy == ChunkStrategy.EXACT:
        chunked_data = chunk_data_exact(source_data, chunk_size, chunk_overlap)
    elif chunk_strategy == ChunkStrategy.SUMMARY:
        chunked_data = summary_chunker(source_data, chunk_size, chunk_overlap)
    else:
        chunked_data = vanilla_chunker(source_data, chunk_size, chunk_overlap)

    return chunked_data


def vanilla_chunker(source_data, chunk_size=100, chunk_overlap=20):
    """Chunk the given source data into smaller parts using a vanilla strategy."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size
                                                   , chunk_overlap=chunk_overlap
                                                   , length_function=len)
    pages = text_splitter.create_documents([source_data])
    return pages


def summary_chunker(source_data, chunk_size=400, chunk_overlap=20):
    """Chunk the given source data into smaller parts, focusing on summarizing content."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size
                                                   , chunk_overlap=chunk_overlap
                                                   , length_function=len)
    try:
        pages = text_splitter.create_documents([source_data])
    except Exception as e:
        pages = text_splitter.create_documents(source_data.content)
        logging.error(f"An error occurred: %s {str(e)}")

    if len(pages) > 10:
        return pages[:5] + pages[-5:]
    return pages


def chunk_data_exact(data_chunks, chunk_size, chunk_overlap):
    """Chunk the data into exact sizes as specified, without considering content."""
    data = "".join(data_chunks)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size - chunk_overlap)]
    return chunks


def chunk_by_sentence(data_chunks, chunk_size, overlap):
    """Chunk the data by sentences, ensuring each chunk does not exceed the specified size."""
    data = "".join(data_chunks)
    sentence_endings = r"(?<=[.!?…]) +"
    sentences = re.split(sentence_endings, data)

    sentence_chunks = []
    for sentence in sentences:
        if len(sentence) > chunk_size:
            chunks = chunk_data_exact([sentence], chunk_size, overlap)
            sentence_chunks.extend(chunks)
        else:
            sentence_chunks.append(sentence)
    return sentence_chunks


def chunk_data_by_paragraph(data_chunks, chunk_size, overlap, bound=0.75):
    """Chunk the data by paragraphs, with consideration for chunk size and overlap."""
    data = "".join(data_chunks)
    total_length = len(data)
    chunks = []
    check_bound = int(bound * chunk_size)
    start_idx = 0

    while start_idx < total_length:
        end_idx = min(start_idx + chunk_size, total_length)
        next_paragraph_index = data.find("\n\n", start_idx + check_bound, end_idx)

        if next_paragraph_index != -1:
            end_idx = next_paragraph_index + 2

        chunks.append(data[start_idx:end_idx + overlap])
        start_idx = end_idx

    return chunks
