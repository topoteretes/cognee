from enum import Enum
from typing import List

from pydantic import BaseModel


class TextSubclass(str, Enum):
    SOURCE_CODE = "Source code in various programming languages"
    SHELL_SCRIPTS = "Shell commands and scripts"
    MARKUP_LANGUAGES = "Markup languages (HTML, XML)"
    STYLESHEETS = "Stylesheets (CSS) and configuration files (YAML, JSON, INI)"
    OTHER = "Other that does not fit into any of the above categories"


class ContentType(BaseModel):
    """Base class for content type, storing type of content as string."""

    type: str = "TEXT"


class TextContent(ContentType):
    """Textual content class for more specific text categories."""

    subclass: List[TextSubclass]


class CodeContentPrediction(BaseModel):
    """Model to predict the type of content."""

    label: TextContent
