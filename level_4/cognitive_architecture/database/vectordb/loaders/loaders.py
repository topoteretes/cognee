from io import BytesIO
import fitz
import os
import sys

from cognitive_architecture.database.vectordb.chunkers.chunkers import chunk_data

from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import DirectoryLoader
import logging
import os
from langchain.document_loaders import TextLoader
import requests
async def _document_loader( observation: str, loader_settings: dict):

    document_format = loader_settings.get("format", "text")
    loader_strategy = loader_settings.get("strategy", "VANILLA")
    chunk_size = loader_settings.get("chunk_size", 500)
    chunk_overlap = loader_settings.get("chunk_overlap", 20)


    logging.info("LOADER SETTINGS %s", loader_settings)

    list_of_docs = loader_settings["path"]
    chunked_doc = []

    if loader_settings.get("source") == "URL":
        for file in list_of_docs:
            if document_format == "PDF":
                logging.info("File is %s", file)
                pdf_response = requests.get(file)
                pdf_stream = BytesIO(pdf_response.content)
                with fitz.open(stream=pdf_stream, filetype='pdf') as doc:
                    file_content = ""
                    for page in doc:
                        file_content += page.get_text()
                pages = chunk_data(chunk_strategy=loader_strategy, source_data=file_content, chunk_size=chunk_size,
                                   chunk_overlap=chunk_overlap)

                chunked_doc.append(pages)

            elif document_format == "TEXT":
                loader = UnstructuredURLLoader(urls=file)
                file_content = loader.load()
                pages = chunk_data(chunk_strategy=loader_strategy, source_data=file_content, chunk_size=chunk_size,
                                   chunk_overlap=chunk_overlap)
                chunked_doc.append(pages)

    elif loader_settings.get("source") == "DEVICE":

        current_directory = os.getcwd()
        logging.info("Current Directory: %s", current_directory)

        loader = DirectoryLoader(".data", recursive=True)
        if document_format == "PDF":
            # loader = SimpleDirectoryReader(".data", recursive=True, exclude_hidden=True)
            documents = loader.load()
            pages = chunk_data(chunk_strategy=loader_strategy, source_data=str(documents), chunk_size=chunk_size,
                               chunk_overlap=chunk_overlap)
            logging.info("Documents: %s", documents)
            # pages = documents.load_and_split()
            chunked_doc.append(pages)


        elif document_format == "TEXT":
            documents = loader.load()
            pages = chunk_data(chunk_strategy=loader_strategy, source_data=str(documents), chunk_size=chunk_size,
                               chunk_overlap=chunk_overlap)
            logging.info("Documents: %s", documents)
            # pages = documents.load_and_split()
            chunked_doc.append(pages)

    else:
        raise ValueError(f"Error: ")
    return chunked_doc





