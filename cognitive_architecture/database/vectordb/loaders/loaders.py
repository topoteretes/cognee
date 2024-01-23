from io import BytesIO
import fitz
import os
import sys

from cognitive_architecture.database.vectordb.chunkers.chunkers import chunk_data
from cognitive_architecture.shared.language_processing import translate_text, detect_language

from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import DirectoryLoader
import logging
import os
from langchain.document_loaders import TextLoader
import requests


async def fetch_pdf_content(file_url):
    response =  requests.get(file_url)
    pdf_stream = BytesIO(response.content)
    with fitz.open(stream=pdf_stream, filetype='pdf') as doc:
        return "".join(page.get_text() for page in doc)

async def fetch_text_content(file_url):
    loader = UnstructuredURLLoader(urls=file_url)
    return loader.load()

async def process_content(content, metadata,  loader_strategy, chunk_size, chunk_overlap):
    pages = chunk_data(chunk_strategy=loader_strategy, source_data=content, chunk_size=chunk_size,
                       chunk_overlap=chunk_overlap)

    if metadata is None:
        metadata = {"metadata": "None"}

    chunk_count= 0

    for chunk in pages:
        chunk_count+=1
        chunk.metadata = metadata
        chunk.metadata["chunk_count"]=chunk_count
    if detect_language(pages) != "en":
        logging.info("Translating Page")
        for page in pages:
            if detect_language(page.page_content) != "en":
                page.page_content = translate_text(page.page_content)

    return pages

async def _document_loader(observation: str, loader_settings: dict):
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
                content = await fetch_pdf_content(file)
            elif document_format == "TEXT":
                content = await fetch_text_content(file)
            else:
                raise ValueError(f"Unsupported document format: {document_format}")

            pages = await process_content(content, metadata=None, loader_strategy=loader_strategy, chunk_size= chunk_size, chunk_overlap= chunk_overlap)
            chunked_doc.append(pages)

    elif loader_settings.get("source") == "DEVICE":
        if loader_settings.get("bulk_load", False) == True:
            current_directory = os.getcwd()
            logging.info("Current Directory: %s", current_directory)
            loader = DirectoryLoader(".data", recursive=True)
            documents = loader.load()
            for document in documents:
                # print ("Document: ", document.page_content)
                pages = await process_content(content= str(document.page_content), metadata=document.metadata, loader_strategy= loader_strategy, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
                chunked_doc.append(pages)
        else:
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(loader_settings.get("single_document_path"))
            documents= loader.load()

            for document in documents:
                pages = await process_content(content=str(document.page_content), metadata=document.metadata,
                                              loader_strategy=loader_strategy, chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
                chunked_doc.append(pages)
    else:
        raise ValueError(f"Unsupported source type: {loader_settings.get('source')}")

    return chunked_doc



# async def _document_loader( observation: str, loader_settings: dict):
#
#     document_format = loader_settings.get("format", "text")
#     loader_strategy = loader_settings.get("strategy", "VANILLA")
#     chunk_size = loader_settings.get("chunk_size", 500)
#     chunk_overlap = loader_settings.get("chunk_overlap", 20)
#
#
#     logging.info("LOADER SETTINGS %s", loader_settings)
#
#     list_of_docs = loader_settings["path"]
#     chunked_doc = []
#
#     if loader_settings.get("source") == "URL":
#         for file in list_of_docs:
#             if document_format == "PDF":
#                 logging.info("File is %s", file)
#                 pdf_response = requests.get(file)
#                 pdf_stream = BytesIO(pdf_response.content)
#                 with fitz.open(stream=pdf_stream, filetype='pdf') as doc:
#                     file_content = ""
#                     for page in doc:
#                         file_content += page.get_text()
#                 pages = chunk_data(chunk_strategy=loader_strategy, source_data=file_content, chunk_size=chunk_size,
#                                    chunk_overlap=chunk_overlap)
#                 from cognitive_architecture.shared.language_processing import translate_text,detect_language
#
#                 if detect_language(pages) != "en":
#                     logging.info("Current Directory 3")
#                     for page in pages:
#                         if detect_language(page.page_content) != "en":
#                             logging.info("Translating Page")
#                             page.page_content = translate_text(page.page_content)
#
#                     chunked_doc.append(pages)
#
#                     logging.info("Document translation complete. Proceeding...")
#
#                 chunked_doc.append(pages)
#
#             elif document_format == "TEXT":
#                 loader = UnstructuredURLLoader(urls=file)
#                 file_content = loader.load()
#                 pages = chunk_data(chunk_strategy=loader_strategy, source_data=file_content, chunk_size=chunk_size,
#                                    chunk_overlap=chunk_overlap)
#
#                 from cognitive_architecture.shared.language_processing import translate_text, detect_language
#
#                 if detect_language(pages) != "en":
#                     logging.info("Current Directory 3")
#                     for page in pages:
#                         if detect_language(page.page_content) != "en":
#                             logging.info("Translating Page")
#                             page.page_content = translate_text(page.page_content)
#
#                     chunked_doc.append(pages)
#
#                     logging.info("Document translation complete. Proceeding...")
#
#                 chunked_doc.append(pages)
#
#     elif loader_settings.get("source") == "DEVICE":
#
#         current_directory = os.getcwd()
#         logging.info("Current Directory: %s", current_directory)
#
#         loader = DirectoryLoader(".data", recursive=True)
#         if document_format == "PDF":
#             # loader = SimpleDirectoryReader(".data", recursive=True, exclude_hidden=True)
#             documents = loader.load()
#             pages = chunk_data(chunk_strategy=loader_strategy, source_data=str(documents), chunk_size=chunk_size,
#                                chunk_overlap=chunk_overlap)
#             logging.info("Documents: %s", documents)
#             from cognitive_architecture.shared.language_processing import translate_text, detect_language
#
#             if detect_language(pages) != "en":
#                 logging.info("Current Directory 3")
#                 for page in pages:
#                     if detect_language(page.page_content) != "en":
#                         logging.info("Translating Page")
#                         page.page_content = translate_text(page.page_content)
#
#                 chunked_doc.append(pages)
#
#                 logging.info("Document translation complete. Proceeding...")
#
#             # pages = documents.load_and_split()
#             chunked_doc.append(pages)
#
#
#         elif document_format == "TEXT":
#             documents = loader.load()
#             pages = chunk_data(chunk_strategy=loader_strategy, source_data=str(documents), chunk_size=chunk_size,
#                                chunk_overlap=chunk_overlap)
#             logging.info("Documents: %s", documents)
#             # pages = documents.load_and_split()
#             chunked_doc.append(pages)
#
#     else:
#         raise ValueError(f"Error: ")
#     return chunked_doc








