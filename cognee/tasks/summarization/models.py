from typing import Union

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models import DocumentChunk
from cognee.shared.CodeGraphEntities import CodeFile, CodePart


class TextSummary(DataPoint):
    """
    Represent a text summary derived from a document chunk.

    This class encapsulates a text summary as well as its associated metadata. The public
    instance variables include 'text' for the summary content and 'made_from' which
    indicates the source document chunk. The 'metadata' instance variable contains
    additional information such as indexed fields.
    """

    text: str
    made_from: DocumentChunk

    metadata: dict = {"index_fields": ["text"]}


class CodeSummary(DataPoint):
    """
    Summarizes code and its components.

    This class inherits from DataPoint and contains a text representation alongside the
    summarized content, which can either be a full code file or a part of it. The metadata
    dictionary defines index fields for the class's instances, particularly focusing on the
    'text' attribute. Public attributes include 'text', 'summarizes', and 'metadata'.
    """

    text: str
    summarizes: Union[CodeFile, CodePart]

    metadata: dict = {"index_fields": ["text"]}
