from langchain.document_loaders import PyPDFLoader
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared.chunk_strategy import ChunkStrategy
import re
def chunk_data(chunk_strategy=None, source_data=None, chunk_size=None, chunk_overlap=None):

    if chunk_strategy == ChunkStrategy.VANILLA:
        chunked_data = vanilla_chunker(source_data, chunk_size, chunk_overlap)

    elif chunk_strategy == ChunkStrategy.PARAGRAPH:
        chunked_data = chunk_data_by_paragraph(source_data,chunk_size, chunk_overlap)

    elif chunk_strategy == ChunkStrategy.SENTENCE:
        chunked_data = chunk_by_sentence(source_data, chunk_size, chunk_overlap)
    elif chunk_strategy == ChunkStrategy.EXACT:
        chunked_data = chunk_data_exact(source_data, chunk_size, chunk_overlap)
    else:
        chunked_data = vanilla_chunker(source_data, chunk_size, chunk_overlap)
    return chunked_data


def vanilla_chunker(source_data, chunk_size=100, chunk_overlap=20):
    # adapt this for different chunking strategies
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    try:
        pages = text_splitter.create_documents([source_data])
    except:
        pages = text_splitter.create_documents(source_data.content)
    # pages = source_data.load_and_split()
    return pages
def chunk_data_exact(data_chunks, chunk_size, chunk_overlap):
    data = "".join(data_chunks)
    chunks = []
    for i in range(0, len(data), chunk_size - chunk_overlap):
        chunks.append(data[i:i + chunk_size])
    return chunks


def chunk_by_sentence(data_chunks, chunk_size, overlap):
    # Split by periods, question marks, exclamation marks, and ellipses
    data = "".join(data_chunks)

    # The regular expression is used to find series of charaters that end with one the following chaacters (. ! ? ...)
    sentence_endings = r'(?<=[.!?…]) +'
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
    data = "".join(data_chunks)
    total_length = len(data)
    chunks = []
    check_bound = int(bound * chunk_size)
    start_idx = 0

    while start_idx < total_length:
        # Set the end index to the minimum of start_idx + default_chunk_size or total_length
        end_idx = min(start_idx + chunk_size, total_length)

        # Find the next paragraph index within the current chunk and bound
        next_paragraph_index = data.find('\n\n', start_idx + check_bound, end_idx)

        # If a next paragraph index is found within the current chunk
        if next_paragraph_index != -1:
            # Update end_idx to include the paragraph delimiter
            end_idx = next_paragraph_index + 2

        chunks.append(data[start_idx:end_idx + overlap])

        # Update start_idx to be the current end_idx
        start_idx = end_idx

    return chunks