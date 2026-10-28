from llama_index.core import Document
from llama_index.core.schema import ImageDocument

from cognee.modules.ingestion import StoredFile, save_data_to_file_detailed
from cognee.modules.ingestion.exceptions import IngestionError


async def get_data_from_llama_index(data_point: Document | ImageDocument) -> str:
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
    stored = await get_data_from_llama_index_detailed(data_point)
    return stored.file_path


async def get_data_from_llama_index_detailed(data_point: Document | ImageDocument) -> StoredFile:
    """Preserve whether a LlamaIndex item was stored or handed through as a path."""
    # Exact types preserve the existing contract for supported LlamaIndex inputs.
    if type(data_point) is Document:
        file_path = data_point.metadata.get("file_path")
    elif type(data_point) is ImageDocument:
        file_path = data_point.image_path
    else:
        raise IngestionError(message=f"Data type not supported: {type(data_point)}")

    if file_path is None:
        return await save_data_to_file_detailed(data_point.text)
    return StoredFile(file_path=file_path)
