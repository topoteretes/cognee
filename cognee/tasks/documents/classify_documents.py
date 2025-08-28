from cognee.modules.data.models import Data
import json
from cognee.modules.data.processing.document_types import (
    Document,
    PdfDocument,
    AudioDocument,
    ImageDocument,
    TextDocument,
    UnstructuredDocument,
)
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from cognee.tasks.documents.exceptions import WrongDataDocumentInputError

EXTENSION_TO_DOCUMENT_CLASS = {
    "pdf": PdfDocument,  # Text documents
    "txt": TextDocument,
    "docx": UnstructuredDocument,
    "doc": UnstructuredDocument,
    "odt": UnstructuredDocument,
    "xls": UnstructuredDocument,
    "xlsx": UnstructuredDocument,
    "ppt": UnstructuredDocument,
    "pptx": UnstructuredDocument,
    "odp": UnstructuredDocument,
    "ods": UnstructuredDocument,
    "png": ImageDocument,  # Image documents
    "dwg": ImageDocument,
    "xcf": ImageDocument,
    "jpg": ImageDocument,
    "jpx": ImageDocument,
    "apng": ImageDocument,
    "gif": ImageDocument,
    "webp": ImageDocument,
    "cr2": ImageDocument,
    "tif": ImageDocument,
    "bmp": ImageDocument,
    "jxr": ImageDocument,
    "psd": ImageDocument,
    "ico": ImageDocument,
    "heic": ImageDocument,
    "avif": ImageDocument,
    "aac": AudioDocument,  # Audio documents
    "mid": AudioDocument,
    "mp3": AudioDocument,
    "m4a": AudioDocument,
    "ogg": AudioDocument,
    "flac": AudioDocument,
    "wav": AudioDocument,
    "amr": AudioDocument,
    "aiff": AudioDocument,
}


def update_node_set(document):
    """
    Extracts node_set from document's external_metadata.

    Parses the external_metadata of the given document and updates the document's
    belongs_to_set attribute with NodeSet objects generated from the node_set found in the
    external_metadata. If the external_metadata is not valid JSON, is not a dictionary, does
    not contain the 'node_set' key, or if node_set is not a list, the function has no effect
    and will return early.

    Parameters:
    -----------

        - document: The document object which contains external_metadata from which the
          node_set will be extracted.
    """
    try:
        external_metadata = json.loads(document.external_metadata)
    except json.JSONDecodeError:
        return

    if not isinstance(external_metadata, dict):
        return

    if "node_set" not in external_metadata:
        return

    node_set = external_metadata["node_set"]
    if not isinstance(node_set, list):
        return

    document.belongs_to_set = [
        NodeSet(id=generate_node_id(f"NodeSet:{node_set_name}"), name=node_set_name)
        for node_set_name in node_set
    ]


async def classify_documents(data_documents: list[Data]) -> list[Document]:
    """
    Classifies a list of data items into specific document types based on their file
    extensions.

    This function processes each item in the provided list of data documents, retrieves
    relevant metadata, and creates instances of document classes mapped to their extensions.
    It ensures that the data items are valid before performing the classification and
    invokes `update_node_set` to extract and set relevant node information from the
    document's external metadata.

    Parameters:
    -----------

        - data_documents (list[Data]): A list of Data objects representing the documents to
          be classified.

    Returns:
    --------

        - list[Document]: A list of Document objects created based on the classified data
          documents.
    """
    if not isinstance(data_documents, list):
        raise WrongDataDocumentInputError("data_documents")

    documents = []
    for data_item in data_documents:
        document = EXTENSION_TO_DOCUMENT_CLASS[data_item.extension](
            id=data_item.id,
            title=f"{data_item.name}.{data_item.extension}",
            raw_data_location=data_item.raw_data_location,
            name=data_item.name,
            mime_type=data_item.mime_type,
            external_metadata=json.dumps(data_item.external_metadata, indent=4),
        )
        update_node_set(document)
        documents.append(document)

    return documents
