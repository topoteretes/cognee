from io import BytesIO
import fitz
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from level_3.vectordb.chunkers.chunkers import chunk_data
from llama_hub.file.base import SimpleDirectoryReader

import requests
def _document_loader( observation: str, loader_settings: dict):
    # Check the format of the document
    document_format = loader_settings.get("format", "text")
    loader_strategy = loader_settings.get("strategy", "VANILLA")
    chunk_size = loader_settings.get("chunk_size", 500)
    chunk_overlap = loader_settings.get("chunk_overlap", 20)

    print("LOADER SETTINGS", loader_settings)

    if document_format == "PDF":
        if loader_settings.get("source") == "URL":
            pdf_response = requests.get(loader_settings["path"])
            pdf_stream = BytesIO(pdf_response.content)
            with fitz.open(stream=pdf_stream, filetype='pdf') as doc:
                file_content = ""
                for page in doc:
                    file_content += page.get_text()
            pages = chunk_data(chunk_strategy= loader_strategy, source_data=file_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            return pages
        elif loader_settings.get("source") == "file":

            loader = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True)
            documents = loader.load_data()
            pages = documents.load_and_split()
            return pages

    elif document_format == "text":
        pages = chunk_data(chunk_strategy= loader_strategy, source_data=observation, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return pages

    else:
        raise ValueError(f"Unsupported document format: {document_format}")


