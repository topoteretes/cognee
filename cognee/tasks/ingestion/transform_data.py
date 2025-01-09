from llama_index.core import Document
from llama_index.core.schema import ImageDocument
from cognee.modules.ingestion import save_data_to_file
from typing import Union


def get_data_from_llama_index(data_point: Union[Document, ImageDocument], dataset_name: str) -> str:
    # Specific type checking is used to ensure it's not a child class from Document
    if isinstance(data_point, Document) and type(data_point) is Document:
        file_path = data_point.metadata.get("file_path")
        if file_path is None:
            file_path = save_data_to_file(data_point.text)
            return file_path
        return file_path
    elif isinstance(data_point, ImageDocument) and type(data_point) is ImageDocument:
        if data_point.image_path is None:
            file_path = save_data_to_file(data_point.text)
            return file_path
        return data_point.image_path
