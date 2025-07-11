from llama_index.core import Document
from llama_index.core.schema import ImageDocument
from cognee.modules.ingestion import save_data_to_file
from typing import Union


async def get_data_from_llama_index(data_point: Union[Document, ImageDocument]) -> str:
    """
    Retrieve the file path based on the data point type.

    Ensure the data point is an instance of either Document or ImageDocument. If the data
    point has a metadata or image path file path, return it; otherwise, save the data
    point's text to a file and return the newly created file path.

    Parameters:
    -----------

        - data_point (Union[Document, ImageDocument]): An instance of Document or
          ImageDocument to extract data from.

    Returns:
    --------

        - str: The file path as a string where the data is stored or the existing path from
          the data point.
    """
    # Specific type checking is used to ensure it's not a child class from Document
    if isinstance(data_point, Document) and type(data_point) is Document:
        file_path = data_point.metadata.get("file_path")
        if file_path is None:
            file_path = await save_data_to_file(data_point.text)
            return file_path
        return file_path
    elif isinstance(data_point, ImageDocument) and type(data_point) is ImageDocument:
        if data_point.image_path is None:
            file_path = await save_data_to_file(data_point.text)
            return file_path
        return data_point.image_path
